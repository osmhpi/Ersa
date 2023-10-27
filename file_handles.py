#!/usr/bin/env python
def print_file_handles():
    import psutil

    proc = psutil.Process()
    file_handles = proc.open_files()
    parend_file_handles = proc.parent().open_files()
    actual_file_handles = set(file_handles) - set(parend_file_handles)
    print(actual_file_handles)


with open("LICENSE", "r") as fp:
    print_file_handles()
