echo "Downloading dogs datas..."
curl "https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/v4-dataset/dogImages.zip" > "dogImages.zip"

echo "Unziping dogs data.."
unzip -o dogImages.zip

echo "Downloading faces data..."
curl "https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/v4-dataset/lfw.zip" > "lfw.zip"

echo "Unziping faces data..."
unzip -o lfw.zip 

echo "Downloading bottleneck-weights..."
mkdir bottleneck_features
curl "https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/v4-dataset/DogVGG16Data.npz" > "bottleneck_features/DogVGG16Data.npz"

echo "All done!"

rm *.zip
