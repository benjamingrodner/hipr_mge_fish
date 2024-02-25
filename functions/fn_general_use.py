import os

def check_dirs(path):
    # split by dirs
    normalized_path = os.path.normpath(path)
    path_components = normalized_path.split(os.sep)
    # Go through dirs in path
    dir = '.'
    ext = os.path.splitext(path)[1]
    path_components = path_components[:-1] if ext else path_components
    for d in path_components:
        dir += '/' + d
        # create the dir if it doesn't exits
        if not os.path.exists(dir):
            # If dir doesnt exist, make dir
            os.makedirs(dir)
            print('Made dir: ', dir)
    return
