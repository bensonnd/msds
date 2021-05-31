import os


# # Return all files in dir, and all its subdirectories, ending in pattern
# def gen_files(dir, pattern):
#     for dirname, _, files in os.walk(dir):
#         for f in files:
#             if f.endswith(pattern):
#                 yield os.path.join(dirname, f)


# # Remove all files in the current dir matching *.config
# for f in gen_files(".", "Zone.Identifier"):
#     os.remove(f)

path = "/home/bensonnd/msds/ds7333/case_study_3/Data2/easy_ham"
all_files = os.listdir(path)

files_to_keep = all_files[:20]

for f in all_files:
    if f not in files_to_keep:
        os.remove(f)
