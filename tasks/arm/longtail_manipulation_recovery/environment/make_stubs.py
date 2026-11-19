"""Create an empty model.xml for every catalogue object instance folder that is missing (see the Dockerfile)."""
import os, sys
listing, root = sys.argv[1], sys.argv[2]
made = 0
for rel in open(listing).read().split():
    d = os.path.join(root, rel)
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "model.xml")
    if not os.path.exists(f):
        open(f, "w").close()
        made += 1
print("stub object instances:", made)
