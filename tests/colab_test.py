from google.colab import drive
import shutil
import IPython.utils.coloransi as coloransi
if not hasattr(coloransi.TermColors, 'Green'):
    coloransi.TermColors.Green = getattr(coloransi.TermColors, 'green', '\033[32m')

# Your original code can now safely execute below
from google.colab import drive

# 1. Mount Google Drive to your runtime notebook session
drive.mount('/content/drive')

# 2. Ship the compressed file straight to your target Drive folder
# shutil.copy('my_data_backup.tar.gz', '/content/drive/MyDrive/data_engineering_backups/')
