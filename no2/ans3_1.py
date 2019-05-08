# -*- coding: utf-8 -*-
import numpy as np

class PolynomialModel():
    def calc_w(self,x_dash,y_dash):
        inv_x_dash = np.linalg.inv(x_dash)  
        return np.dot(inv_x_dash,y_dash)

    def calc_estimate_value(self, x, w):
        return np.dot(x, w)
        
    def calc_generalization_error(self,y_dash_dash,y_estimated_odd):
        return 0.5 * np.linalg.norm(abs(y_dash_dash - y_estimated_odd))


if __name__ == "__main__":
    # initialize
    x = np.array([[1, 169, np.power(169, 2), np.power(169, 3)],
    [1, 183, np.power(183, 2), np.power(183, 3)],
    [1, 156, np.power(156, 2), np.power(156, 3)],
    [1, 176, np.power(176, 2), np.power(176, 3)],
    [1, 165, np.power(165, 2), np.power(165, 3)],
    [1, 166, np.power(166, 2), np.power(166, 3)],
    [1, 172, np.power(172, 2), np.power(172, 3)],
    [1, 171, np.power(171, 2), np.power(171, 3)],
    ])
    
    x_dash = np.array([[1, 169, np.power(169, 2), np.power(169, 3)],
    [1, 156, np.power(156, 2), np.power(156, 3)],
    [1, 165, np.power(165, 2), np.power(165, 3)],
    [1, 172, np.power(172, 2), np.power(172, 3)]
    ])
    y = np.array([[56], [70], [52], [77], [68], [56], [71], [61]])
    y_dash = np.array([[56], [52], [68], [71]])
    
    report3_1 = PolynomialModel()
    w = report3_1.calc_w(x_dash, y_dash)
    print("【no.1】")
    print("===w===")
    print(w)

    print("===y-estimated===")
    estimate_value = report3_1.calc_estimate_value(x, w)
    print(estimate_value)

    y_dash_dash = np.array([[70], [77], [56], [61]])
    y_estimated_odd = np.array([estimate_value[1],estimate_value[3],[estimate_value[5]],estimate_value[7]])
    
    print("===Generalization Error===")
    generalization_error = report3_1.calc_generalization_error(y_dash_dash, y_estimated_odd)
    print(generalization_error)

