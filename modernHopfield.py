import numpy as np
import Hopfield


def transform_to_bipolar(unipolar_image):
    """
    :param unipolar_image: unipolar 2d numpy array
    :return: bipolar image of shape unipolar_image.shape
    """
    return unipolar_image.astype(np.int32)*2-1


def transform_to_unipolar(bipolar_image):
    """
    :param bipolar_image: bipolar 2d numpy array
    :return: unipolar image of shape bipolar_image.shape
    """
    return (bipolar_image.astype(np.int32) - 1) // 2


class ModernHopfield:
    def __init__(self):
        self.X = Hopfield.get_x('ready_leaves')
        self.X = transform_to_bipolar(self.X)

    def process_image(self, image, is_bipolar=False, return_bipolar=False):
        """
        Let the image to process by Modern Hopfield network
        :param image: flatten 1d image
        :param is_bipolar: True is image is bipolar - False otherwise
        :param return_bipolar: if True network will return bipolar image,
                otherwise returns unipolar image
        :return: Processed image
        """
        if not is_bipolar:
            eps = transform_to_bipolar(image)
        else:
            eps = np.copy(image, dtype=np.int32)
        print(eps)
        while True:
            print("Iteration")
            new_eps = self.calc_new_eps(eps)
            if np.all(new_eps == eps):
                break
            else:
                eps = new_eps
        if not return_bipolar:
            return transform_to_unipolar(eps)
        else:
            return eps

    def calc_new_eps(self, old_eps):
        E_pos = np.zeros(old_eps.shape)
        E_neg = np.zeros(old_eps.shape)
        for i in range(old_eps.shape[0]):
            eps_plus = np.copy(old_eps)
            eps_plus[i] = 1
            eps_minus = np.copy(old_eps)
            eps_minus[i] = -1
            for j in range(self.X.shape[0]):
                plus = self.X[j].T @ eps_plus
                minus = self.X[j].T @ eps_minus
                E_pos[i] = E_pos[i] + np.exp(plus/10)
                E_neg[i] = E_neg[i] + np.exp(minus/10)

        new_eps = np.sign(E_pos-E_neg)
        return new_eps
