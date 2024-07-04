import click
import tflite
import numpy as np

from layer import *

# TODO, Add output layer name to manually stop at some layer
@click.command()
@click.argument('path', type =click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('uuid', type = click.INT)
@click.argument('appid', type = click.INT)
@click.argument('export', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('-t', '--test', type = click.BOOL)
def lf_generate_snapshot_cmd(path, uuid, appid, export, test):
    return lf_generate_snapshot(path, uuid, appid, export, test)

def lf_generate_snapshot(path, uuid, appid, export, test):
    click.echo("Reading model from %s ..." % click.format_filename(path))
    if not path.endswith('.tflite'):
        click.echo("The file should end with .tflite")
        return

    with open(path, 'rb') as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)

    graph = model.Subgraphs(0)
    num_ops = graph.OperatorsLength()
    layer_list = []
    extra_include_list = []
    for op_index in range(0, num_ops):
        op = graph.Operators(op_index)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        if op_code.BuiltinCode() == tflite.BuiltinOperator.FULLY_CONNECTED:
            assert(op.InputsLength() == 3)
            input_tensor, input_buffer = get_tensor_and_buffer(model, graph, op.Inputs(0))
            #print("input_tensor: ", input_tensor)
            weight_tensor, weight_buffer = get_tensor_and_buffer(model, graph, op.Inputs(1))
            #print("weight_tensor: ", weight_tensor)
            
            input_tensor_shape = [input_tensor.Shape(i) for i in range(input_tensor.ShapeLength())]
            weight_tensor_shape = [weight_tensor.Shape(i) for i in range(weight_tensor.ShapeLength())]

            print("input_tensor shape: ", input_tensor_shape)
            print("weight_tensor shape: ", weight_tensor_shape)

            if input_tensor_shape[1] != weight_tensor_shape[1]:
                print(f"Padding input_tensor from shape {input_tensor_shape} to match weight_tensor shape {weight_tensor_shape}")
                padded_input_shape = [input_tensor_shape[0], weight_tensor_shape[1]]
                
                if input_buffer is None:
                    print("input_buffer is NOne, initializing new buffer")
                    input_buffer = np.zeros((input_tensor_shape[0], input_tensor_shape[1]), dtype=np.float32)

                print("Original input_buffer shape:", input_buffer.shape)
                if input_buffer.ndim == 1:
                    input_buffer = np.expand_dims(input_buffer.shape)
                input_buffer = np.pad(input_buffer, ((0, 0), (0, weight_tensor_shape[1] - input_tensor_shape[1])), 'constant', constant_values=0)
                print("Padded input_buffer shape:",input_buffer.shape)

                input_tensor_shape = padded_input_shape
                print("Updated input_tensor_shape after padding:", input_tensor_shape)

            print("Final input_tensor.Shape: ", input_tensor_shape)
            print("Final weight_tensor.Shape: ", weight_tensor_shape)

            
            bias_tensor, bias_buffer = get_tensor_and_buffer(model, graph, op.Inputs(2))
            
            output_tensor, output_buffer = get_tensor_and_buffer(model, graph, op.Outputs(0))
            print("input_tensor shape[1]: ", input_tensor.Shape(1))
            print("weight_tensor shape[1]: ", weight_tensor.Shape(1))
            input_tensor_shape = [input_tensor.Shape(i) for i in range(input_tensor.ShapeLength())]
            weight_tensor_shape = [weight_tensor.Shape(i) for i in range(weight_tensor.ShapeLength())]
            print("input_tensor shape: ", input_tensor_shape)
            print("weight_tensor shape: ", weight_tensor_shape)
            
#            if input_tensor_shape == [1, 1]:
#                padded_input_shape = [1, 30]
#                input_buffer = np.pad(input_buffer, ((0,29)), 'constant', constant_values=0)
#                print("padded_input_buffer shape: ", input_buffer.shape)

            layer = FCLayer(op_code, input_tensor, weight_tensor, bias_tensor, output_tensor,
                                        input_buffer, weight_buffer, bias_buffer, output_buffer)
            layer_list.append(layer)

        elif op_code.BuiltinCode() == tflite.BuiltinOperator.TANH:
            assert(op.InputsLength() == 1)
            assert(op.OutputsLength() == 1)
            input_tensor, input_buffer = get_tensor_and_buffer(model, graph, op.Inputs(0))
            output_tensor, output_buffer = get_tensor_and_buffer(model, graph, op.Outputs(0))

            layer = TanhLayer(op_code, input_tensor, output_tensor, input_buffer, output_buffer)
            layer_list.append(layer)
            
            if 'tanh_lookup_table.h' not in extra_include_list:
                extra_include_list.append('tanh_lookup_table.h')

        
        elif op_code.BuiltinCode() == tflite.BuiltinOperator.QUANTIZE:
            assert(op.InputsLength() == 1)
            assert(op.OutputsLength() == 1)
            input_tensor, input_buffer = get_tensor_and_buffer(model, graph, op.Inputs(0))
            output_tensor, output_buffer = get_tensor_and_buffer(model, graph, op.Outputs(0))

            layer = QuanLayer(op_code, input_tensor, output_tensor, input_buffer, output_buffer)
            layer_list.append(layer)

        elif op_code.BuiltinCode() == tflite.BuiltinOperator.DEQUANTIZE:
            assert(op.InputsLength() == 1)
            assert(op.OutputsLength() == 1)
            input_tensor, input_buffer = get_tensor_and_buffer(model, graph, op.Inputs(0))
            output_tensor, output_buffer = get_tensor_and_buffer(model, graph, op.Outputs(0))

            layer = DeQuanLayer(op_code, input_tensor, output_tensor, input_buffer, output_buffer)
            layer_list.append(layer)

        else:
            click.echo("Unsupported OP Code: %s ..." % op_code.BuiltinCode())
            continue

    model_input_size = layer_list[0].input_size
    model_output_size = layer_list[-1].output_size

    TEMPLATE_FILE = "main.c"
    _template = template.get_template(TEMPLATE_FILE)
    code = _template.render(model_uuid = uuid,
                            app_id = appid,
                            layer_list = layer_list,
                            input_size = model_input_size,
                            output_size = model_output_size,
                            extra_include_list = extra_include_list,
                            test_mode = True)

    OUTPUT_FILE = f"{export}/lf_model_{uuid}.c"
    with open(OUTPUT_FILE, "w") as output_file:
        output_file.write(code)

def get_tensor_and_buffer(model, graph, input):
    tensor = graph.Tensors(input)
    
    tensor_type = tensor.Type()
    print("current tensor type: ", tensor_type)
    print("tflite.TensorType.FLOAT32: ", tflite.TensorType.FLOAT32)
    print("tflite.TensorType.INT8: ", tflite.TensorType.INT8)
    print("tflite.TensorType.INT32: ", tflite.TensorType.INT32)
    #실행하기 위해 임의로 추가한 조건문
    print("tflite.TensorType.UINT8: ", tflite.TensorType.UINT8)

    raw_buffer = model.Buffers(tensor.Buffer()).DataAsNumpy()

    if tensor_type == tflite.TensorType.FLOAT32:
        viewer = '<f4'		
    elif tensor_type == tflite.TensorType.INT8:
        viewer = '<i1'		
    elif tensor_type == tflite.TensorType.INT32:
        viewer = '<i4'
    # 실행하기 위해 임의로 추가한 조건문
    elif tensor_type == tflite.TensorType.UINT8:
        viewer = '<u1'
    else:
        raise Exception('Unsupported Tensor Type: %s ...' % tensor_type)
    
    tensor_shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
    print("Tensor shape: ", tensor_shape)

    if isinstance(raw_buffer, np.ndarray) and raw_buffer.size > 0:
        buffer = raw_buffer.view(viewer)
    else:
        buffer = None

    # 버퍼가 None인 경우 초기화
    #if buffer is None:
    #    buffer = np.zeros(tensor.ShapeAsNumpy(), dtype=viewer)
    #    print(f"Initialized buffer with shape: {tensor.ShapeAsNumpy()} and type: {viewer}")

    # 텐서의 shape 출력
    tensor_shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
    print("Tensor shape: ", tensor_shape)

    return tensor, buffer

if __name__ == "__main__":
    lf_generate_snapshot_cmd()
