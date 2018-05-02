/*
pom.xml
==============================
        <dependency>
            <groupId>javaxt</groupId>
            <artifactId>javaxt-core</artifactId>
            <version>1.8.0</version>
        </dependency>

        <dependency>
            <groupId>com.yesup.oss</groupId>
            <artifactId>tensorflow-client</artifactId>
            <version>1.4-2</version>
        </dependency>

        <dependency>
            <groupId>io.grpc</groupId>
            <artifactId>grpc-netty</artifactId>
            <version>1.7.0</version>
        </dependency>
*/

public static void main(String[] args) {
        String modelName = "faster_rcnn_resnet101_pets";
        String signatureName = "serving_default";

        try {

            String file = "/Users/liangyanfeng/workspace/models/research/object_detection/test_images/image1.jpg";

            BufferedImage image = new javaxt.io.Image(new File(file)).getBufferedImage();

            List<Integer> intList = new ArrayList<>();
            int pixels[] = image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());
            // RGB转BGR格式
            for(int i=0,j=0;i<pixels.length;++i,j+=3){
                intList.add(pixels[i] & 0xff);
                intList.add((pixels[i] >> 8) & 0xff);
                intList.add((pixels[i] >> 16) & 0xff);
            }

            //记个时
            long t = System.currentTimeMillis();
            //创建连接，注意usePlaintext设置为true表示用非SSL连接
            ManagedChannel channel = ManagedChannelBuilder.forAddress("0.0.0.0", 9000).usePlaintext(true).build();

            //这里还是先用block模式
            PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
            //创建请求
            Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();

            //模型名称和模型方法名预设
            Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
            modelSpecBuilder.setName(modelName);
            modelSpecBuilder.setSignatureName(signatureName);
            predictRequestBuilder.setModelSpec(modelSpecBuilder);


            //设置入参,访问默认是最新版本，如果需要特定版本可以使用tensorProtoBuilder.setVersionNumber方法
            TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
            tensorProtoBuilder.setDtype(DataType.DT_UINT8);

            TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(image.getHeight()));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(image.getWidth()));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));

            tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());

            tensorProtoBuilder.addAllIntVal(intList);

            predictRequestBuilder.putInputs("inputs", tensorProtoBuilder.build());

            //访问并获取结果
            Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());

            List<Float> boxes = predictResponse.getOutputsOrThrow("detection_boxes").getFloatValList();
            List<Float> scores = predictResponse.getOutputsOrThrow("detection_scores").getFloatValList();
            List<Float> classes = predictResponse.getOutputsOrThrow("detection_classes").getFloatValList();
            System.out.println(boxes);
            System.out.println(scores);
            System.out.println(classes);

            System.out.println("cost time: " + (System.currentTimeMillis() - t));


        } catch (Exception e){
            e.printStackTrace();
        }
    }
