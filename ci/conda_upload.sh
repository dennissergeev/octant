PKG_NAME=octant
USER=dennissergeev

OS=$TRAVIS_OS_NAME-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=`python -c 'import octant; print(octant.__version__)'`
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l nightly --force $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION.tar.bz2
