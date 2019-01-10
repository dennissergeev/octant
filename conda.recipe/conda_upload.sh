PKG_NAME=octant
USER=dennissergeev

OS=$TRAVIS_OS_NAME-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=`python -c 'import octant; print(octant.__version__)'`
conda build --no-test .
PKG_FULL_NAME=`conda build --output .`
# echo 'PKG_FULL_NAME='$PKG_FULL_NAME
if [[ $VERSION == *"dirty"* ]]; then
    LABEL="nightly";
else
    LABEL="main";
fi;
echo "label: $LABEL"
echo ""
anaconda --token $CONDA_UPLOAD_TOKEN upload -u $USER -l $LABEL --force $PKG_FULL_NAME
