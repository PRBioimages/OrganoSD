import cellpose.models as cp_model
import cellpose.resnet_torch as cp_net
import torch
import os
import argparse


def convert_to_ONNX(model_path: str, output_directory: str, diam_mean: float, batch_size: int = 1):
    residual_on, style_on, concatenation = True, True, False
    try:
        model = cp_net.CPnet(
            [2, 32, 64, 128, 256],
            3,
            sz=3,
            mkldnn=None,
            diam_mean=diam_mean)
        model.load_model(model_path)
        model.eval()

        # convert to onnx
        onnx_model_path = os.path.join(output_directory, os.path.split(model_path)[-1] + ".onnx")
        dummy = torch.randn(batch_size, 2, 512, 512, requires_grad=True)

        torch.onnx.export(
            model,
            dummy,
            onnx_model_path,
            verbose=False,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output', 'style'])
    except:
        print(f"Couldn't convert {model_path}.")


def convert_all_models(output_directory: str):
    # get built-in model names and custom model names
    all_models = cp_model.MODEL_NAMES.copy()
    model_strings = cp_model.get_user_models()
    all_models.extend(model_strings)

    for model_type in all_models:
        model_string = model_type if model_type is not None else 'cyto'
        if model_string == 'nuclei':
            diam_mean = 17.
        else:
            diam_mean = 30.

        if model_type == 'cyto' or model_type == 'cyto2' or model_type == 'nuclei':
            model_range = range(4)
        else:
            model_range = range(1)

        model_pathes = [cp_model.model_path(model_string, j) for j in model_range]

        for model_path in model_pathes:
            convert_to_ONNX(model_path, output_directory, diam_mean)


def main():

    import time
    time.sleep(20)
    parser = argparse.ArgumentParser(description='Converter parameters')
    parser.add_argument('--output_directory', default="/home/hryang/Detecseg/cyto3",
                        type=str, help='Output directory for converted models. ')
    parser.add_argument('--model_path', default="/home/hryang/Detecseg/cyto3.pth", type=str,
                        help='full path to the individual cellpose model')
    parser.add_argument('--mean_diameter', default=17.0, type=float,
                          help='Mean diameter used for training the given model. 17.0 for nuclei-based models, otherwise 30.0')

    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    if args.model_path:
        if 'mean_diameter' not in vars(args) or args.mean_diameter is None:
            raise Exception("The __model_path argument requires the __mean_diameter.")
        if args.mean_diameter != 17.0 and args.mean_diameter != 30.0:
            raise Exception("Mean_diameter must be either 17.0 (for nuclei-based models) or 30.0 for all other models.")
        convert_to_ONNX(model_path=args.model_path, output_directory=args.output_directory, diam_mean=args.mean_diameter)
    else:
        # get user models and built-in models
        convert_all_models(args.output_directory)

    print("Output models are saved here: ", args.output_directory)
    print("Conversion completed.")


if __name__ == "__main__":
    main()
