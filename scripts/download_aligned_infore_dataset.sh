data_root="./train_data" # modify this
pushd .
mkdir -p $data_root
cd $data_root
gdown 1iRYyAQP9A_afd3MRhceGPqI532UGSvnm -O aligned.zip
unzip -q aligned.zip 
popd
