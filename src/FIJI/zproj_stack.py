from ij import IJ, ImagePlus, WindowManager
from ij.plugin.frame import RoiManager
from ij.process import ImageStatistics as ImageStatistics
from ij.gui import WaitForUserDialog, Toolbar
from ij.io import DirectoryChooser
from ij.plugin import ZProjector, ImagesToStack


from loci.plugins import BF

import os
import fnmatch

dc = DirectoryChooser("pick folder")
directory = dc.getDirectory()
os.chdir(directory) #select data folder in PyCharm project
#subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

for filename in os.listdir(directory):
	#print(fnmatch.fnmatch(filename, '*registered*.tif'))
	if fnmatch.fnmatch(filename, '*slice*.tif') and not fnmatch.fnmatch(filename, '*ROIs.tif'):
		print('VVVVfound .tif file: enterVVVV')
		print(os.path.join(directory, filename))
		#imps = IJ.openImage(os.path.join(directory, folder, filename))
		imps = BF.openImagePlus(os.path.join(directory, filename))
		slices = 0
		for imp in imps:
			imp.show()
			slices+=1
		imp_zproj = ZProjector.run (imp, 'avg')
		imp_zproj.show()
stack = ImagesToStack()
stack.convertImagesToStack()
rm = RoiManager()
#roi = RoiManager.getRoiManager()
roi = RoiManager.getInstance() 