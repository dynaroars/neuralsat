from neuralsat.test import reset_settings, extract_instance
from neuralsat import InteractiveVerifier, Verifier
from neuralsat.util.misc.logger import logger
import logging

if __name__ == "__main__":

    ############## Preprocess ##############

    logger.setLevel(logging.INFO)
    reset_settings()

    net_path = 'src/neuralsat/example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'src/neuralsat/example/vnnlib/prop_1_0.03.vnnlib'
    device = 'cpu'
    batch = 1
    print(f'\n\nRunning test with {net_path=} {vnnlib_path=}')

    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)

    verifier = Verifier(
        net=model,
        input_shape=input_shape,
        batch=batch,
        device=device,
    )
    
    status = verifier.verify(objectives)
    print(f'{status=}')
    
    # env = InteractiveVerifier(
    #     net=model,
    #     input_shape=input_shape,
    #     batch=batch,
    #     device=device,
    # )

    # objectives, _ = env._preprocess(objectives, force_split=None)
    # objective = objectives.pop(1)

    # ############## Verify ##############


    # done = env.init(
    #     objective=objective,
    #     reference_bounds=None,
    #     preconditions={},
    # )
    
    
    # name_dict = {
    #     i: node.name for i, node in enumerate(env.abstractor.net.split_nodes)
    # }
    
    # print(f'{name_dict=}')

    # while not done:
    #     observation, subproblems = env.get_observation(batch)
    #     action, _state = env.decide(observation, subproblems)
    #     reward, done, info = env.step(action)
    #     print(action[0])
    #     # for i in range(len(observation)):
    #     #     for j in range(len(observation[i])):
    #     #         print(f'observation[{i=}][{j=}]={observation[i][j].shape}')
        
    #     # (topk, batch)
    #     all_rewards, all_decisions = env.get_rewards(subproblems)
    #     print(f'{all_rewards.shape=}')
    #     print(f'{all_decisions=}')
    #     exit()
