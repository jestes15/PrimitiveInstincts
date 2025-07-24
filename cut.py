# header-begin ------------------------------------------
# File       : cut.py
#
# Author      : Joshua E
# Email       : estesjn2020@gmail.com
#
# Created on  : 7/23/2025
#
# header-end --------------------------------------------

string = """amd-smi-lib amdgpu-core comgr composablekernel-dev gcc-11-base gdal-data gdal-plugins half hip-dev hip-doc hip-runtime-amd hip-samples hipblas hipblas-common-dev hipblas-dev hipblaslt hipblaslt-dev hipcc hipcub-dev hipfft hipfft-dev hipfort-dev
  hipify-clang hiprand hiprand-dev hipsolver hipsolver-dev hipsparse hipsparse-dev hipsparselt hipsparselt-dev hiptensor hiptensor-dev hsa-amd-aqlprofile hsa-rocr hsa-rocr-dev libaec0 libamd-comgr2 libamd3 libamdhip64-5 libarmadillo12 libarpack2t64
  libasan6 libavcodec-dev libavformat-dev libavutil-dev libblosc1 libcamd3 libccolamd3 libcfitsio10t64 libcharls2 libcholmod5 libcolamd3 libdc1394-25 libdc1394-dev libdrm-amdgpu-amdgpu1 libdrm-amdgpu-common libdrm-amdgpu-dev libdrm-amdgpu-radeon1
  libdrm-dev libdrm2-amdgpu libelf-dev libevent-pthreads-2.1-7t64 libexif-dev libexif-doc libexif12 libfabric1 libfile-copy-recursive-perl libfile-listing-perl libfile-which-perl libfreexl1 libfyba0t64 libgcc-11-dev libgdal34t64 libgdcm-dev
  libgdcm3.0t64 libgeos-c1t64 libgeos3.12.1t64 libgeotiff5 libgl2ps1.4 libglew2.2 libgphoto2-6t64 libgphoto2-dev libgphoto2-l10n libgphoto2-port12t64 libhdf4-0-alt libhdf5-103-1t64 libhdf5-hl-100t64 libhsa-runtime64-1 libhsakmt1 libhttp-date-perl
  libhwloc-plugins libhwloc15 libimath-3-1-29t64 libimath-dev libkmlbase1t64 libkmldom1t64 libkmlengine1t64 liblept5 libllvm17t64 libmunge2 libmysqlclient21 libnetcdf19t64 libnuma-dev libodbc2 libodbcinst2 libogdi4.1 libopencv-calib3d-dev
  libopencv-calib3d406t64 libopencv-contrib-dev libopencv-contrib406t64 libopencv-core-dev libopencv-core406t64 libopencv-dnn-dev libopencv-dnn406t64 libopencv-features2d-dev libopencv-features2d406t64 libopencv-flann-dev libopencv-flann406t64
  libopencv-highgui-dev libopencv-highgui406t64 libopencv-imgcodecs-dev libopencv-imgcodecs406t64 libopencv-imgproc-dev libopencv-imgproc406t64 libopencv-java libopencv-ml-dev libopencv-ml406t64 libopencv-objdetect-dev libopencv-objdetect406t64
  libopencv-photo-dev libopencv-photo406t64 libopencv-shape-dev libopencv-shape406t64 libopencv-stitching-dev libopencv-stitching406t64 libopencv-superres-dev libopencv-superres406t64 libopencv-video-dev libopencv-video406t64 libopencv-videoio-dev
  libopencv-videoio406t64 libopencv-videostab-dev libopencv-videostab406t64 libopencv-viz-dev libopencv-viz406t64 libopencv406-jni libopenexr-3-1-30 libopenexr-dev libopenmpi3t64 libpciaccess-dev libpmix2t64 libpoppler134 libpq5 libproj25
  libprotobuf32t64 libpsm-infinipath1 libpsm2-2 libqhull-r8.0 libqt5core5t64 libqt5dbus5t64 libqt5gui5t64 libqt5network5t64 libqt5opengl5t64 libqt5qml5 libqt5qmlmodels5 libqt5quick5 libqt5svg5 libqt5test5t64 libqt5waylandclient5 libqt5waylandcompositor5
  libqt5widgets5t64 libraw1394-11 libraw1394-dev libraw1394-tools librdmacm1t64 librttopo1 libsocket++1 libspatialite8t64 libstdc++-11-dev libsuitesparseconfig7 libsuperlu6 libswresample-dev libswscale-dev libsz2 libtbb-dev libtbb12 libtbbbind-2-5
  libtbbmalloc2 libtesseract5 libtimedate-perl libtsan0 libucx0 liburi-perl liburiparser1 libusb-1.0-0 libvtk9.1t64 libxerces-c3.2t64 libxnvctrl0 mesa-common-dev migraphx migraphx-dev miopen-hip miopen-hip-dev mivisionx mysql-common opencv-data
  openmp-extras-dev openmp-extras-runtime poppler-data proj-bin proj-data python3-argcomplete python3-pip qt5-gtk-platformtheme qttranslations5-l10n qtwayland5 rccl rccl-dev rocalution rocalution-dev rocblas rocblas-dev rocfft rocfft-dev rocm-cmake
  rocm-core rocm-dbgapi rocm-debug-agent rocm-developer-tools rocm-device-libs rocm-gdb rocm-hip-libraries rocm-hip-runtime rocm-hip-runtime-dev rocm-hip-sdk rocm-language-runtime rocm-llvm rocm-ml-libraries rocm-ml-sdk rocm-opencl rocm-opencl-dev
  rocm-opencl-runtime rocm-opencl-sdk rocm-openmp-sdk rocm-smi-lib rocm-utils rocminfo rocprim-dev rocprofiler rocprofiler-compute rocprofiler-dev rocprofiler-plugins rocprofiler-register rocprofiler-sdk rocprofiler-sdk-roctx rocprofiler-systems rocrand
  rocrand-dev rocsolver rocsolver-dev rocsparse rocsparse-dev rocthrust-dev roctracer roctracer-dev rocwmma-dev rpp rpp-dev unixodbc-common valgrind"""

output = string.replace("\n", "").split(" ")

for package in output:
    if package.find("opencv") == -1:
        print(f"{package}", end=" ")

# footer-begin ------------------------------------------
# default.Python
# File       : cut.py
# footer-end --------------------------------------------