# VD-01 Microcontroller boards

This page is an excerpt of data that the vendors themselves publish (product pages, datasheets, drawings): one
vendor's family of microcontroller boards that share a board outline and a header layout. The values are the vendors'
own, in the vendors' units; nothing is converted. "not published" marks a value that the vendor pages used for this
excerpt do not state. The page sets comparable parts side by side; it does not say which of them, if any, is on the
robot.

## Arduino A000005

| field | value |
|---|---|
| name | Arduino(R) Nano |
| part number(s) listed | A000005 |
| part number with header strips fitted | not published |
| microcontroller | ATmega328 |
| USB connector | Mini-B USB |
| length x width | 45 mm x 18 mm |
| Wi-Fi | not published |
| Bluetooth | not published |
| IMU in the technical specification | not published |
| IMU chips named anywhere in the vendor's documents of the board | none |
| header connectors | not published |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | PB5 D13 D12 PB4 CIPO |
| labels of the row at the far end of the drawing | VIN D1/TX PD1 |

Header pins, as the vendor's pin table numbers them (table "Analog"):

| pin | function |
|---|---|
| 1 | +3V3 |
| 2 | A0 |
| 3 | A1 |
| 4 | A2 |
| 5 | A3 |
| 6 | A4 |
| 7 | A5 |
| 8 | A6 |
| 9 | A7 |
| 10 | +5V |
| 11 | Reset |
| 12 | GND |
| 12 | VIN |

Header pins, as the vendor's pin table numbers them (table "Digital"):

| pin | function |
|---|---|
| 1 | D1/TX1 |
| 2 | D0/RX0 |
| 3 | D2 |
| 4 | D3 |
| 5 | D4 |
| 6 | D5 |
| 7 | D6 |
| 8 | D7 |
| 9 | D8 |
| 10 | D9 |
| 11 | D10 |
| 12 | D11 |
| 13 | D12 |
| 14 | D13 |
| 15 | Reset |
| 16 | GND |

## Arduino ABX00027

| field | value |
|---|---|
| name | Arduino(R) Nano 33 IoT |
| part number(s) listed | ABX00027 |
| part number with header strips fitted | ABX00032 |
| microcontroller | SAMD21 Cortex(R)-M0+ 32bit low power ARM MCU |
| USB connector | Micro USB |
| length x width | 45 mm x 18 mm |
| Wi-Fi | Nina W102 uBlox module |
| Bluetooth | Nina W102 uBlox module |
| IMU in the technical specification | LSM6DS3 |
| IMU chips named anywhere in the vendor's documents of the board | LSM6DS3, LSM6DSL, LSM6DSOXTR |
| header connectors | 15 pins per connector (two connectors) |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK (SC1) PA17 D13 ~D12 PA19 CIPO (SC1) |
| labels of the row at the far end of the drawing | VIN TX PB22 |

Header pins, as the vendor's pin table numbers them (table "Headers"):

| pin | function |
|---|---|
| 1 | D13 |
| 2 | +3V3 |
| 3 | AREF |
| 4 | A0/DAC0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4/SDA |
| 9 | A5/SCL |
| 10 | A6 |
| 11 | A7 |
| 12 | VUSB |
| 13 | RST |
| 14 | GND |
| 15 | VIN |
| 16 | TX |
| 17 | RX |
| 18 | RST |
| 19 | GND |
| 20 | D2 |
| 21 | D3/PWM |
| 22 | D4 |
| 23 | D5/PWM |
| 24 | D6/PWM |
| 25 | D7 |
| 26 | D8 |
| 27 | D9/PWM |
| 28 | D10/PWM |
| 29 | D11/MOSI |
| 30 | D12/MISO |

## Arduino ABX00028

| field | value |
|---|---|
| name | Arduino(R) Nano Every |
| part number(s) listed | ABX00028 |
| part number with header strips fitted | ABX00033 |
| microcontroller | ATMega4809 |
| USB connector | Micro USB |
| length x width | 45 mm x 18 mm |
| Wi-Fi | not published |
| Bluetooth | not published |
| IMU in the technical specification | not published |
| IMU chips named anywhere in the vendor's documents of the board | LSM6DSL |
| header connectors | 15 pins per connector (two connectors) |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK PE2 D13 D12 PE1 CIPO (SC1) |
| labels of the row at the far end of the drawing | VIN TX PC4 |

Header pins, as the vendor's pin table numbers them (table "Headers"):

| pin | function |
|---|---|
| 1 | D13 |
| 2 | +3V3 |
| 3 | AREF |
| 4 | A0/DAC0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4/SDA |
| 9 | A5/SCL |
| 10 | A6 |
| 11 | A7 |
| 12 | +5V |
| 13 | RST |
| 14 | GND |
| 15 | VIN |
| 16 | TX |
| 17 | RX |
| 18 | RST |
| 19 | GND |
| 20 | D2 |
| 21 | D3/PWM |
| 22 | D4 |
| 23 | D5/PWM |
| 24 | D6/PWM |
| 25 | D7 |
| 26 | D8 |
| 27 | D9/PWM |
| 28 | D10/PWM |
| 29 | D11/MOSI |
| 30 | D12/MISO |

## Arduino ABX00030

| field | value |
|---|---|
| name | Arduino(R) Nano 33 BLE |
| part number(s) listed | ABX00030 |
| part number with header strips fitted | ABX00034 |
| microcontroller | nRF52840 |
| USB connector | Micro USB |
| length x width | 45 mm x 18 mm |
| Wi-Fi | not published |
| Bluetooth | NINA-B306 |
| IMU in the technical specification | LSM9DS1 |
| IMU chips named anywhere in the vendor's documents of the board | LSM6DSL, LSM9DS1 |
| header connectors | 15 pins per connector (two connectors) |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK P0.13 D13~ ~D12 P1.08 CIPO |
| labels of the row at the far end of the drawing | VIN TX P1.03 |

Header pins, as the vendor's pin table numbers them (table "Headers"):

| pin | function |
|---|---|
| 1 | D13 |
| 2 | +3V3 |
| 3 | AREF |
| 4 | A0/DAC0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4/SDA |
| 9 | A5/SCL |
| 10 | A6 |
| 11 | A7 |
| 12 | VUSB |
| 13 | RST |
| 14 | GND |
| 15 | VIN |
| 16 | TX |
| 17 | RX |
| 18 | RST |
| 19 | GND |
| 20 | D2 |
| 21 | D3/PWM |
| 22 | D4 |
| 23 | D5/PWM |
| 24 | D6/PWM |
| 25 | D7 |
| 26 | D8 |
| 27 | D9/PWM |
| 28 | D10/PWM |
| 29 | D11/MOSI |
| 30 | D12/MISO |

## Arduino ABX00031

| field | value |
|---|---|
| name | Arduino(R) Nano 33 BLE Sense |
| part number(s) listed | ABX00031 |
| part number with header strips fitted | not published |
| microcontroller | nRF52840 |
| USB connector | Micro USB |
| length x width | 45 mm x 18 mm |
| Wi-Fi | not published |
| Bluetooth | NINA-B306 |
| IMU in the technical specification | LSM9DS |
| IMU chips named anywhere in the vendor's documents of the board | LSM6DSL, LSM9DS, LSM9DS1 |
| header connectors | 15 pins per connector (two connectors) |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK P0.13 D13~ ~D12 P1.08 CIPO |
| labels of the row at the far end of the drawing | VIN TX P1.03 |

Header pins, as the vendor's pin table numbers them (table "Headers"):

| pin | function |
|---|---|
| 1 | D13 |
| 2 | +3V3 |
| 3 | AREF |
| 4 | A0/DAC0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4/SDA |
| 9 | A5/SCL |
| 10 | A6 |
| 11 | A7 |
| 12 | VUSB |
| 13 | RST |
| 14 | GND |
| 15 | VIN |
| 16 | TX |
| 17 | RX |
| 18 | RST |
| 19 | GND |
| 20 | D2 |
| 21 | D3/PWM |
| 22 | D4 |
| 23 | D5/PWM |
| 24 | D6/PWM |
| 25 | D7 |
| 26 | D8 |
| 27 | D9/PWM |
| 28 | D10/PWM |
| 29 | D11/MOSI |
| 30 | D12/MISO |

## Arduino ABX00053

| field | value |
|---|---|
| name | Arduino(R) Nano RP2040 Connect |
| part number(s) listed | ABX00053 |
| part number with header strips fitted | not published |
| microcontroller | Raspberry Pi RP2040 |
| USB connector | Micro USB |
| length x width | 45 mm x 18 mm |
| Wi-Fi | Nina W102 uBlox module |
| Bluetooth | Nina W102 uBlox module |
| IMU in the technical specification | LSM6DSOXTR (6-axis) |
| IMU chips named anywhere in the vendor's documents of the board | LSM6DSOX, LSM6DSOXTR |
| header connectors | not published |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK GPIO6 D13 D12 GPIO4 CIPO |
| labels of the row at the far end of the drawing | VIN TX GPIO0 |

Header pins, as the vendor's pin table numbers them (table "JP1"):

| pin | function |
|---|---|
| 1 | TX1 |
| 2 | RX0 |
| 3 | RST |
| 4 | GND |
| 5 | D2 |
| 6 | D3 |
| 7 | D4 |
| 8 | D5 |
| 9 | D6 |
| 10 | D7 |
| 11 | D8 |
| 12 | D9 |
| 13 | D10 |
| 14 | D11 |
| 15 | D12 |

Header pins, as the vendor's pin table numbers them (table "JP2"):

| pin | function |
|---|---|
| 1 | D13 |
| 2 | 3.3V |
| 3 | REF |
| 4 | A0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4 |
| 9 | A5 |
| 10 | A6 |
| 11 | A7 |
| 12 | VUSB |
| 13 | REC |
| 14 | GND |
| 15 | VIN |

## Arduino ABX00069

| field | value |
|---|---|
| name | Arduino(R) Nano 33 BLE Sense Rev2 |
| part number(s) listed | ABX00069 |
| part number with header strips fitted | not published |
| microcontroller | nRF52840 |
| USB connector | Micro USB |
| length x width | 45 mm x 18 mm |
| Wi-Fi | not published |
| Bluetooth | NINA-B306 |
| IMU in the technical specification | BMI270 (3-axis accelerometer + 3-axis gyroscope) + BMM150 (3-axis Magnetometer) |
| IMU chips named anywhere in the vendor's documents of the board | BMI270, BMM150, LSM9DS1 |
| header connectors | 15 pins per connector (two connectors) |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK P0.13 D13~ ~D12 P1.08 CIPO |
| labels of the row at the far end of the drawing | VIN D0/TX P1.03 |

Header pins, as the vendor's pin table numbers them (table "Headers"):

| pin | function |
|---|---|
| 1 | D13 |
| 2 | +3V3 |
| 3 | AREF |
| 4 | A0/DAC0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4/SDA |
| 9 | A5/SCL |
| 10 | A6 |
| 11 | A7 |
| 12 | VUSB |
| 13 | RST |
| 14 | GND |
| 15 | VIN |
| 16 | TX |
| 17 | RX |
| 18 | RST |
| 19 | GND |
| 20 | D2 |
| 21 | D3/PWM |
| 22 | D4 |
| 23 | D5/PWM |
| 24 | D6/PWM |
| 25 | D7 |
| 26 | D8 |
| 27 | D9/PWM |
| 28 | D10/PWM |
| 29 | D11/MOSI |
| 30 | D12/MISO |

## Arduino ABX00071 / ABX00072

| field | value |
|---|---|
| name | Arduino(R) Nano 33 BLE Rev2 |
| part number(s) listed | ABX00071 / ABX00072 |
| part number with header strips fitted | not published |
| microcontroller | nRF52840 |
| USB connector | Micro USB |
| length x width | 45 mm x 18 mm |
| Wi-Fi | not published |
| Bluetooth | NINA-B306 |
| IMU in the technical specification | BMI270 (3-axis accelerometer + 3-axis gyroscope) + BMM150 (3-axis Magnetometer) |
| IMU chips named anywhere in the vendor's documents of the board | BMI270, BMM150 |
| header connectors | 15 pins per connector (two connectors) |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK P0.13 ~D13 ~D12 P1.08 CIPO |
| labels of the row at the far end of the drawing | VIN ~D0/TX P1.03 TX0 |

Header pins, as the vendor's pin table numbers them (table "Headers"):

| pin | function |
|---|---|
| 1 | D13 |
| 2 | +3V3 |
| 3 | AREF |
| 4 | A0/DAC0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4/SDA |
| 9 | A5/SCL |
| 10 | A6 |
| 11 | A7 |
| 12 | VUSB |
| 13 | RST |
| 14 | GND |
| 15 | VIN |
| 16 | TX |
| 17 | RX |
| 18 | RST |
| 19 | GND |
| 20 | D2 |
| 21 | D3/PWM |
| 22 | D4 |
| 23 | D5/PWM |
| 24 | D6/PWM |
| 25 | D7 |
| 26 | D8 |
| 27 | D9/PWM |
| 28 | D10/PWM |
| 29 | D11/MOSI |
| 30 | D12/MISO |

## Arduino ABX00083 / ABX00092 / ABX00083_CN / ABX00092_CN

| field | value |
|---|---|
| name | Arduino(R) Nano ESP32 |
| part number(s) listed | ABX00083 / ABX00092 / ABX00083_CN / ABX00092_CN |
| part number with header strips fitted | ABX00083 |
| microcontroller | u-blox(R) NORA-W106-10B (ESP32-S3) |
| USB connector | USB-C(R) |
| length x width | 45 mm x 18 mm |
| Wi-Fi | Wi-Fi 4 (802.11b/g/n) 2.4 GHz |
| Bluetooth | Bluetooth 5.0 (BLE support) |
| IMU in the technical specification | not published |
| IMU chips named anywhere in the vendor's documents of the board | none |
| header connectors | not published |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK GPIO48 ~D13 ~D12 GPIO47 CIPO |
| labels of the row at the far end of the drawing | VIN ~D1/TX0 GPIO43 TX0 |

Header pins, as the vendor's pin table numbers them (table "Analog (JP1)"):

| pin | function |
|---|---|
| 1 | D13 / SCK |
| 2 | +3V3 |
| 3 | BOOT0 |
| 4 | A0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4 |
| 9 | A5 |
| 10 | A6 |
| 11 | A7 |
| 12 | VBUS |
| 13 | BOOT1 |
| 14 | GND |
| 15 | VIN |

Header pins, as the vendor's pin table numbers them (table "Digital (JP2)"):

| pin | function |
|---|---|
| 1 | D12 / CIPO* |
| 2 | D11 / COPI* |
| 3 | D10 / CS* |
| 4 | D9 |
| 5 | D8 |
| 6 | D7 |
| 7 | D6 |
| 8 | D5 |
| 9 | D4 |
| 10 | D3 |
| 11 | D2 |
| 12 | GND |
| 13 | RST |
| 14 | D0/RX |
| 15 | D1/TX |

## Arduino ABX00112 (Castellated Pins) / ABX00137 (With Headers)

| field | value |
|---|---|
| name | Arduino(R) Nano Matter |
| part number(s) listed | ABX00112 (Castellated Pins) / ABX00137 (With Headers) |
| part number with header strips fitted | ABX00137 |
| microcontroller | MGM240SD22VNA 32-bit Arm(R) Cortex(R)-M33 with DSP instruction and FPU |
| USB connector | USB-C(R) |
| length x width | 45 mm x 18 mm |
| Wi-Fi | not published |
| Bluetooth | Bluetooth(R) Low Energy 5.3 & Bluetooth(R) Mesh |
| IMU in the technical specification | not published |
| IMU chips named anywhere in the vendor's documents of the board | none |
| header connectors | not published |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK ~D13 ~D12 MISO |
| labels of the row at the far end of the drawing | not published |

Header pins: the vendor gives no pin table in text.

## Arduino ABX00142 (Castellated Pins) / ABX00143 (With Headers)

| field | value |
|---|---|
| name | Arduino(R) Nano R4 |
| part number(s) listed | ABX00142 (Castellated Pins) / ABX00143 (With Headers) |
| part number with header strips fitted | ABX00143 |
| microcontroller | Renesas RA4M1 (Arm(R) Cortex(R)-M4) |
| USB connector | not published |
| length x width | 45 mm x 18 mm |
| Wi-Fi | not published |
| Bluetooth | not published |
| IMU in the technical specification | not published |
| IMU chips named anywhere in the vendor's documents of the board | none |
| header connectors | not published |
| pinout drawing | USB connector drawn at the top edge; the row holding D13 (left column) and D12 (right column) is the row next to it |
| labels of the row at the USB end of the drawing | SCK D13 D12 MISO |
| labels of the row at the far end of the drawing | not published |

Header pins, as the vendor's pin table numbers them (table "Analog (JP1)"):

| pin | function |
|---|---|
| 1 | D13 / SCK |
| 2 | +3V3 |
| 3 | AREF |
| 4 | A0 |
| 5 | A1 |
| 6 | A2 |
| 7 | A3 |
| 8 | A4 |
| 9 | A5 |
| 10 | A6 |
| 11 | A7 |
| 12 | 5V |
| 13 | BOOT |
| 14 | GND |
| 15 | VIN |

Header pins, as the vendor's pin table numbers them (table "Digital (JP2)"):

| pin | function |
|---|---|
| 15 | D12 / CIPO |
| 14 | D11 / COPI |
| 13 | D10 / CS |
| 12 | D9 |
| 11 | D8 |
| 10 | D7 |
| 9 | D6 |
| 8 | D5 |
| 7 | D4 |
| 6 | D3 |
| 5 | D2 |
| 4 | GND |
| 3 | RST |
| 2 | D0/RX |
| 1 | D1/TX |
