"""Mirror selected Isaac 4.5 asset prefixes from NVIDIA's public S3 bucket into a local dir."""
import os, sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET
BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ROOT = "Assets/Isaac/4.5/"
PREFIXES = ["Isaac/Robots/Franka/", "Isaac/Props/KLT_Bin/", "Isaac/Props/YCB/Axis_Aligned/",
            "Isaac/Environments/Simple_Room/", "Isaac/Materials/"]
out = sys.argv[1]
ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
total = 0
for pre in PREFIXES:
    token = None
    while True:
        q = {"list-type": "2", "prefix": ROOT + pre, "max-keys": "1000"}
        if token: q["continuation-token"] = token
        xml = urllib.request.urlopen(BUCKET + "/?" + urllib.parse.urlencode(q), timeout=60).read()
        root = ET.fromstring(xml)
        for c in root.findall("s3:Contents", ns):
            key = c.find("s3:Key", ns).text; size = int(c.find("s3:Size", ns).text)
            if key.endswith("/"): continue
            dst = os.path.join(out, key[len(ROOT):])
            if os.path.exists(dst) and os.path.getsize(dst) == size: continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(BUCKET + "/" + urllib.parse.quote(key), dst); break
                except Exception as e:
                    if attempt == 2: raise
            total += size
        t = root.find("s3:NextContinuationToken", ns)
        token = t.text if t is not None else None
        if not token: break
    print("done", pre, flush=True)
print("downloaded_bytes", total)
