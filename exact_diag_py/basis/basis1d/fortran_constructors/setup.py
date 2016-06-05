def get_sources():
	import os,glob
	package_dir = os.path.dirname(os.path.realpath(__file__))
	sources_dir = os.path.join(package_dir,'sources')
	sources=[]
	for Dir,subDir,files in os.walk(sources_dir):
		src_files=glob.glob(os.path.join(Dir,"*.src"))
		sources.extend(src_files)

	return sources
		
	

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('fortran_constructors',parent_package, top_path)
    config.add_extension('basis_ops', sources=get_sources(),f2py_options=["--quiet"])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())