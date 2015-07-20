import sys
sys.path.append('../../DeepLearningTutorials/code/')
sys.path.append('../clinical_db')

import generate_sample


if __name__ == '__main__':


    if algorithm is 0:
        # logistic regression
        import logistic_sgd
        logistic_sgd.sgd_optimization_mnist()
    elif algorithm is 1:
        # multi-layer perceptron
        import mlp
        mlp.test_mlp()
    elif algorithm is 2:
        # convolutional neural network
        import convolutional_mlp
        convolutional_mlp.evaluate_lenet5()
    elif algorithm is 3:
        # denoising auto-encoder
        import dA
        dA.test_dA()
    elif algorithm is 4:
        # stacked denoising auto-encoders
        import SdA
        SdA.test_SdA()
