import sys

for line in sys.stdin:
    parts = line.split(" ")

    if len(parts) == 2:
        verb, data = parts
        data = int(data) / 2**20
        print(f"{verb} {data:.3f}MiB")
    else:
        print(" ".join(parts))
