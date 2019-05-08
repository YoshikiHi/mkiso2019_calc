from ans3_1 import PolynomialModel
import numpy as np

if __name__ == "__main__":
    # initialize
    x = np.array([[1, 169],[1, 183],[1, 156],[1, 176],[1, 165],[1, 166],[1, 172],[1, 171]])
    x_dash = np.array([[1, 169], [1, 156], [1, 165], [1, 172]])
    y = np.array([[56], [70], [52], [77], [68], [56], [71], [61]])
    y_dash = np.array([[56], [52], [68], [71]])
    
    report3_2 = PolynomialModel()
    w = np.dot(np.dot(np.linalg.inv(np.dot(x_dash.T, x_dash)),x_dash.T),y_dash)
    print("【no.2】")
    print("===w===")
    print(w)

    print("===y-estimated===")
    estimate_value = report3_2.calc_estimate_value(x, w)
    print(estimate_value)

    
    y_dash_dash = np.array([[70], [77], [56], [61]])
    y_estimated_odd = np.array([estimate_value[1], estimate_value[3], [estimate_value[5]], estimate_value[7]])
    print("===Generalization Error===")
    generalization_error = report3_2.calc_generalization_error(y_dash_dash, y_estimated_odd)
    print(generalization_error)