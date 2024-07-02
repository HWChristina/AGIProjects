
# -*- coding: utf-8 -*-

import os
import re
import sys


if __name__ == '__main__':
    if len(sys.argv) !=3:
        print("Usage: python file_rename.py <old_name> <new_name>")
        exit(1)
        
    #获取旧文件名
    
    old_name = sys.argv[1]
    if not os.path.exists(old_name):
        print("File not exists: %s" % old_name)
        exit(1)
        
    #获取新文件名
    new_name = sys.argv[2]
    
    try:
        #重命名
        os.rename(old_name, new_name)
        print("Rename %s to %s" % (old_name, new_name))
    except Exception as e:
        print("Rename failed: %s" % e)
        print(e)
        exit(1)
        
                