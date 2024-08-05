from ij import IJ, ImagePlus, WindowManager
from ij.plugin.frame import RoiManager
from ij.process import ImageStatistics as ImageStatistics
from ij.gui import WaitForUserDialog, Toolbar
from ij.io import DirectoryChooser
from ij.plugin import ZProjector


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
		rm = RoiManager()
		WaitForUserDialog("Select ROIs,then click OK.").show();
		#roi = RoiManager.getRoiManager()
		roi = RoiManager.getInstance() 
#		#roi_points = roi.getRoisAsArray()
#		zip_name = os.path.join(directory, filename.replace('.tif', '_ROIs.zip'))
#		roi.runCommand("Save", zip_name)
#		roi.moveRoisToOverlay(imp)
#		roi.moveRoisToOverlay(imp_zproj)
#		roi.deselect()
#		rt=roi.multiMeasure(imp)
#		rt.getResultsTable()
#		WindowManager.getWindow("results")
#		rt.show("results")
#		csv_name = os.path.join(directory, filename.replace('.tif', '_ROIs.csv'))
#		csv_exist = os.path.exists(csv_name)
#		print("csv file exists:", csv_exist)
#		print("csv file name:", csv_name)
#		if csv_exist:
#			os.remove(csv_name)
#		rt.save(csv_name)
#		IJ.selectWindow("results")
#		IJ.run("Close")
#		#roi.close()
#		imp.close()
#		tif_name = os.path.join(directory, filename.replace('.tif', '_ROIs.tif'))
#		print("saving ROIs as", tif_name)
#		IJ.save(imp_zproj, tif_name)
#		imp_zproj.close()
#		#IJ.selectWindow("AVG_"+filename)
