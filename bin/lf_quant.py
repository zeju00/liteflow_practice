# The tflite file can be further converted into json using
# flatc -t --strict-json --defaults-json  third_party/tensorflow/tensorflow/lite/schema/schema.fbs -- nn-loader/sample/aurora/converted_int_quan_model.tflite 

import click
import tensorflow as tf
import numpy as np


# import for debug
import logging


@click.command()
@click.argument('saved_model_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('dataset_path', type=click.Path(exists=True, file_okay= True, dir_okay=False))
@click.argument('export_path', type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True))
def lf_quant_cmd(saved_model_dir, dataset_path, export_path):
    return lf_quant(saved_model_dir, dataset_path, export_path)

def lf_quant(saved_model_dir, dataset_path, export_path):
    
    print("saved_model_dir: ", saved_model_dir)
    _dataset = np.load(dataset_path).astype(np.float32)
    print("_dataset:\n", _dataset)
    #print("exit")
    #exit(0)

    dataset = tf.data.Dataset.from_tensor_slices(_dataset).batch(1)
    print("dataset:\n", dataset)
    #print("exit")
    #exit(0)

    def representative_data_gen():
        for input_value in dataset.take(60):
            yield [input_value.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    print("tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) success\n")

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    print("optimization success \n")

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    print("target_spec success\n")

    converter.inference_input_type = tf.uint8
    print("input_type success\n")

    converter.inference_output_type = tf.uint8
    print("output_type success\n")

    converter.representative_dataset = representative_data_gen
    print("representative_dataset success\n")
    print(converter.representative_dataset)
    #print("exit")
    #exit(0)

    # print for check data before convert
    #print("before")
    #print(np.load(export_path))
    
    
    tflite_model = converter.convert()
    print("where to save: ", export_path)

    #try:
    #    tflite_model = converter.convert()
    #except Exception as e:
    #    logging.error("Conversion failed: %s", e)
    #    print("exit")
        #exit(0)
        

    # print for check data after convert
    #print("after")
    #print(tflite_model)

    #print("after converting: %s\n", tflite_model)
    open(export_path, "wb").write(tflite_model)

    click.echo("Convert done ...")
    click.echo("Begin inspecting model")

    interpreter = tf.lite.Interpreter(model_path=export_path)
    interpreter.allocate_tensors()
    input_index_quant = interpreter.get_input_details()[0]["index"]
    output_index_quant = interpreter.get_output_details()[0]["index"]

    click.echo("Input index quant: {}, output index quant: {}".format(input_index_quant, output_index_quant))
    
    #print("input index quant type: ", input_index_quant.type(), "output index quant type: ", output_index_quant.type())


    for _data in _dataset:
        #print("data: \n", _data)
        test_data = np.expand_dims(_data, axis=0).astype(np.float32)
        click.echo("Test input: %s\n" % test_data)
        
        # for test
        #test_data_uint8 = (test_data * 255).astype(np.uint8)
        #click.echo("test_data_uint8: %s" % test_data_uint8)
        test_data_uint8 = (test_data * 255).astype(np.uint8)
        print("test_data_uint8:\n{}".format(test_data_uint8))
        interpreter.set_tensor(input_index_quant, test_data_uint8)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index_quant)

        click.echo("Output: %s" % predictions)
    
    print('lf_quant.py 종료')    
    #exit(0)

if __name__ == "__main__":
    print("lf_quant 시작")
    lf_quant_cmd()
