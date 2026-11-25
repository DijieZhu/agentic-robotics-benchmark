"""Telemetry of the robot at 100 Hz (the fields of the robot's own messages): one record per control
period, sampled after the period, kept for the last 30 s; telemetry() returns the records since the previous call
(at most the last 30 s; `dropped` counts the ones lost in between).

Fields (columns of the reply; frames and node numbering in frames.md):
  t                         episode time (s)
  sensor_length_mm[9]       stretch sensors: 0-5 short (parallel to lines 0-5), 6-8 long; white noise 0.5 mm rms,
                            quantised to 0.1 mm (rated value of this robot; no drift)
  sensor_capacitance_pF[9]  C = m L + b of the robot's own calibration, from the reported (noisy) length
  motor_encoder_counts[6]   encoder counts at the output shaft (12 CPR x gear ratio), + = line wound in
  motor_encoder_length_mm[6]  the same as line length on the spool (counts x 2 pi r / counts per turn)
  motor_target_mm[6]        the PID's current target length of each short sensor; null when no command runs
  motor_done[6]             the PID's done flags of the current target state
  motor_duty[6]             PWM duty sent to each motor (-1..1, + = wind in)
  imu_accel_g[3][3]         per strut: accelerometer (specific force, g) in that strut's IMU frame; noise 2 mg rms
  imu_gyro_dps[3][3]        per strut: angular rate (deg/s) in the IMU frame; noise 0.1 deg/s rms plus one constant
                            bias per axis and episode, uniform in [-0.5, 0.5] deg/s
No magnetometer, no power readings (the firmware reads a 6-axis IMU and no power meter).

The noise streams are seeded by the episode's noise seed and advance once per record, so the records do not depend on
when or how often the program calls telemetry().
"""
import math

import numpy as np

RATED = {"length_noise_mm": 0.5, "length_quantum_mm": 0.1, "accel_noise_g": 0.002, "gyro_noise_dps": 0.1,
         "gyro_bias_max_dps": 0.5}
BUFFER_S = 30.0
NOISE_STREAM = 0x7E
COLUMNS = ("t", "sensor_length_mm", "sensor_capacitance_pF", "motor_encoder_counts", "motor_encoder_length_mm",
           "motor_target_mm", "motor_done", "motor_duty", "imu_accel_g", "imu_gyro_dps")
G0 = 9.80665


class Telemetry:
    def __init__(self, S, noise_seed, rate_hz=100.0, buffer_s=BUFFER_S):
        from ysim import params as prm
        self.S = S
        self.n_buf = int(round(buffer_s * rate_hz))
        self.rng = np.random.default_rng([int(noise_seed) & ((1 << 63) - 1), NOISE_STREAM])
        self.gyro_bias = self.rng.uniform(-RATED["gyro_bias_max_dps"], RATED["gyro_bias_max_dps"], (3, 3))
        self.cal_m = np.asarray(S.cal_m, float)
        self.cal_b = np.asarray(S.cal_b, float)
        self.mm_per_count = 2.0 * math.pi * prm.value("spool_radius") * 1000.0 / S.motors[0].cpr_out
        self.min_mm = prm.value("min_length") * 1000.0
        self._sens = np.asarray(S.ids.sensor_tendons, int)
        self._acc = np.asarray([[a + j for j in range(3)] for a in S.ids.acc], int)
        self._gyr = np.asarray([[a + j for j in range(3)] for a in S.ids.gyr], int)
        n = self.n_buf
        self.buf = {"t": np.zeros(n), "len": np.zeros((n, 9)), "cap": np.zeros((n, 9)),
                    "enc": np.zeros((n, 6), np.int64), "encl": np.zeros((n, 6)), "tgt": np.full((n, 6), np.nan),
                    "done": np.zeros((n, 6), bool), "duty": np.zeros((n, 6)), "acc": np.zeros((n, 3, 3)),
                    "gyr": np.zeros((n, 3, 3))}
        self.count = 0           # records written so far
        self.read_upto = 0       # records already returned

    def sample(self, t):
        """One record (called by the cell after every control period). One draw of 27 normals per record: 9 lengths,
        then the 3 x 3 accelerometer and the 3 x 3 gyroscope axes."""
        S, b, i = self.S, self.buf, self.count % self.n_buf
        z = self.rng.standard_normal(27)
        L = S.d.ten_length[self._sens] * 1000.0
        q = RATED["length_quantum_mm"]
        Ln = np.round((L + z[:9] * RATED["length_noise_mm"]) / q) * q
        b["t"][i] = t
        b["len"][i] = Ln
        b["cap"][i] = self.cal_m * Ln + self.cal_b
        for j, mo in enumerate(S.motors):
            b["enc"][i, j] = mo.encoder_counts()
        b["encl"][i] = b["enc"][i] * self.mm_per_count
        ex = S.exec
        if ex is not None and not ex["finished"]:
            st = ex["seq"][ex["k"]]
            b["tgt"][i] = self.min_mm + np.asarray(st["ranges"], float) * np.asarray(st["target"], float)
            b["done"][i] = S.pid.done
        else:
            b["tgt"][i] = np.nan
            b["done"][i] = False
        b["duty"][i] = S.duty
        sd = S.d.sensordata
        b["acc"][i] = sd[self._acc] / G0 + z[9:18].reshape(3, 3) * RATED["accel_noise_g"]
        b["gyr"][i] = np.degrees(sd[self._gyr]) + self.gyro_bias + z[18:27].reshape(3, 3) * RATED["gyro_noise_dps"]
        self.count += 1

    def _rows(self, lo, hi):
        idx = np.arange(lo, hi) % self.n_buf
        return idx

    def read(self):
        """The reply of telemetry(): the records since the previous call, at most the last buffer_s."""
        hi = self.count
        lo = max(self.read_upto, hi - self.n_buf)
        dropped = lo - self.read_upto
        self.read_upto = hi
        idx = self._rows(lo, hi)
        b = self.buf

        def r(a, n):
            return np.round(a, n).tolist()

        tgt = np.round(b["tgt"][idx], 3)
        tgt_l = [[None if not math.isfinite(v) else float(v) for v in row] for row in tgt]
        return {"n": int(len(idx)), "dropped": int(dropped), "columns": list(COLUMNS),
                "t": r(b["t"][idx], 6), "sensor_length_mm": r(b["len"][idx], 1),
                "sensor_capacitance_pF": r(b["cap"][idx], 5), "motor_encoder_counts": b["enc"][idx].tolist(),
                "motor_encoder_length_mm": r(b["encl"][idx], 4), "motor_target_mm": tgt_l,
                "motor_done": b["done"][idx].tolist(), "motor_duty": r(b["duty"][idx], 5),
                "imu_accel_g": r(b["acc"][idx], 5), "imu_gyro_dps": r(b["gyr"][idx], 4)}

    def record(self):
        return {"records": int(self.count), "returned_upto": int(self.read_upto), "rated": dict(RATED),
                "gyro_bias_dps": np.round(self.gyro_bias, 6).tolist()}
