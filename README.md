# falcon-imageprocessing-webapi-sample
This is sample program to create webapi of image processing by falcon python. You can modify this code when you create your own image processing webapi.

## Usage
1. Download [this project](https://github.com/aidiary/keras-examples/tree/master/vgg16/17flowers) and run it.
2. Run below commands.

```
$ pip install cython falcon gunicorn falcon-multipart
$ gunicorn -b 127.0.0.1:5000 debug_falcon:api
```

3. Access webapi by below command.

```
$ curl -X POST http://127.0.0.1:5000/debug -H "Content-Type: multipart/form-data" -F "file=@hoge.jpg"
```
