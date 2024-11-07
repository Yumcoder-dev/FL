import copy
import numpy as np
import nvflare.client as flare


def train(input_arr):
    output_arr = copy.deepcopy(input_arr)
    # mock training with plus 1
    return output_arr + 1


def evaluate(input_arr):
    # mock evaluation metrics
    return np.mean(input_arr)


def main():
    flare.init()

    sys_info = flare.system_info()
    print(f"system info is: {sys_info}", flush=True)

    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")
        print(f"received weights: {input_model.params}")

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}")

        if input_model.params == {}:
            params = np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        else:
            params = np.array(
                input_model.params["numpy_key"], dtype=np.float32)

        # training
        new_params = train(params)

        # evaluation
        metrics = evaluate(params)

        sys_info = flare.system_info()
        print(f"system info is: {sys_info}", flush=True)
        print(f"finished round: {input_model.current_round}", flush=True)

        print(f"sending weights: {new_params}")

        output_model = flare.FLModel(
            params={"numpy_key": new_params},
            params_type="FULL",
            metrics={"accuracy": metrics},
            current_round=input_model.current_round,
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
