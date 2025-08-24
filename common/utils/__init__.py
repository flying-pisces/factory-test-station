from .io_utils import read_line_from_file
from .io_utils import append_results_log_to_csv
from .io_utils import timestamp
from .os_utils import mkdir_p
from .thread_utils import TimeoutThread

# Try to import GUI utilities - they may not work on all platforms
try:
    from .gui_utils import ImageDisplayBox
    from .gui_utils import MessageBox
except ImportError:
    # Provide stubs for platforms where GUI utilities aren't available
    class ImageDisplayBox:
        def __init__(self, *args, **kwargs):
            pass
        def show(self):
            pass
    
    class MessageBox:
        @staticmethod
        def show(*args, **kwargs):
            print("MessageBox:", args, kwargs)
        
        @staticmethod
        def showinfo(*args, **kwargs):
            print("MessageBox Info:", args, kwargs)
        
        @staticmethod
        def showwarning(*args, **kwargs):
            print("MessageBox Warning:", args, kwargs)
        
        @staticmethod
        def showerror(*args, **kwargs):
            print("MessageBox Error:", args, kwargs)
