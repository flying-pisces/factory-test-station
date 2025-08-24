__author__ = 'chuckyin'

# pylint: disable=R0901
# pylint: disable=R0904
# pylint: disable=R0924
# pylint: disable=C0103
# pylint: disable=W0613

import tkinter as tk
#import tkMessageBox
from tkinter import messagebox
import platform

# Windows-specific imports
HAS_WPF = False
if platform.system() == "Windows":
    try:
        import clr
        clr.AddReference('WPFMessageBox')
        clr.AddReference('Hsc')
        from MessageBoxUtils import WPFMessageBox
        from Hsc import ImageDisplayBox as imgDispBox
        HAS_WPF = True
    except ImportError:
        HAS_WPF = False

# Fallback classes for non-Windows platforms
if not HAS_WPF:
    class WPFMessageBox:
        @staticmethod
        def Show(msg, title, msgbtn=0, icon=0):
            # Fallback to console output instead of popup dialogs
            if icon == 16:  # Error
                print(f"ERROR: {title}: {msg}")
            elif icon == 48:  # Warning
                print(f"WARNING: {title}: {msg}")
            else:  # Info
                print(f"INFO: {title}: {msg}")
            return True
    
    class imgDispBox:
        def __init__(self, *args, **kwargs):
            pass
        def show(self):
            pass

class MessageBox(object):
    @classmethod
    def warning(cls, title=None, msg=None, msgbtn=0):
        WPFMessageBox.Show(msg, title, msgbtn, 48)

    @classmethod
    def error(cls, title=None, msg=None, msgbtn=0):
        WPFMessageBox.Show(msg, title, msgbtn, 16)

    @classmethod
    def info(cls, title=None, msg=None, msgbtn=0):
        return WPFMessageBox.Show(msg, title, msgbtn, 64)


class ImageDisplayBox(object):
    @classmethod
    def display(cls, image_file_name):
        img_box = imgDispBox()
        img_box.Show()
        img_box.DisplayImage(image_file_name)
