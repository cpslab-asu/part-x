import numpy as np
import matplotlib.pyplot as plt

def divide_points(samples_in: np.array, samples_out:np.array, region_support: list) -> list:
    """

    Args:
        samples_in: Samples from Training set.
        samples_out: Evaluated values of samples from Training set.
        region_support: Min and Max of all dimensions

    Returns:
        list: Divided samples
    """    
    regionSamples = []
    corresponding_robustenss = []
    if samples_in.shape[0] == samples_out.shape[0] and  samples_out.shape[0] != 0:

        for iterate, subregion in enumerate(region_support):
            boolArray = []
            for dimension in range(len(subregion)):
                subArray = samples_in[:, dimension]
                logical_subArray = np.logical_and(subArray >= subregion[dimension, 0], subArray <= subregion[dimension, 1])
                boolArray.append(np.squeeze(logical_subArray))
            corresponding_robustenss.append(samples_out[(np.all(boolArray, axis = 0))])
            regionSamples.append(samples_in[(np.all(boolArray, axis = 0)),:])
    else:
        for iterate, subregion in enumerate(region_support):
            corresponding_robustenss.append(np.array([]))
            regionSamples.append(np.array([[]]))
            
    return regionSamples, corresponding_robustenss

def testPointInSubRegion(regionSamples, orig_samples, region_support, subregion_support):
    # region_support_1, region_support_2, region_support_3, region_support_4 = subregion_support

    x_coordinates_1 = [region_support[0,0], region_support[0,0]]
    y_coordinates_1 = [region_support[1,0], region_support[1,1]]

    x_coordinates_2 = [region_support[0,0], region_support[0,1]]
    y_coordinates_2 = [region_support[1,0], region_support[1,0]]

    x_coordinates_3 = [region_support[0,1], region_support[0,1]]
    y_coordinates_3 = [region_support[1,0], region_support[1,1]]

    x_coordinates_4 = [region_support[0,0], region_support[0,1]]
    y_coordinates_4 = [region_support[1,1], region_support[1,1]]

    # print(x_coordinates_1, y_coordinates_1)
    # print(x_coordinates_2, y_coordinates_2)
    # print(x_coordinates_3, y_coordinates_3)
    # print(x_coordinates_4, y_coordinates_4)
    plt.plot(x_coordinates_1, y_coordinates_1, color = 'red')
    plt.plot(x_coordinates_2, y_coordinates_2, color = 'blue')
    plt.plot(x_coordinates_3, y_coordinates_3, color = 'green')
    plt.plot(x_coordinates_4, y_coordinates_4, color = 'yellow')

    listStyle = ['b--',  'y--', 'k--', 'g--']
    listStyle_marker = ['b.',  'y.', 'k.', 'g.']
    # subregion_support_1, subregion_support_2, subregion_support_3, subregion_support_4 = subregion_support
    for iterate in range(len(subregion_support)):
        # print(subregion_support[iterate,:,:])
        # print("***********")

        x_coordinates_sub_r_1 = [subregion_support[iterate][0,0], subregion_support[iterate][0,0]]
        y_coordinates_sub_r_1 = [subregion_support[iterate][1,0], subregion_support[iterate][1,1]]

        x_coordinates_sub_r_2 = [subregion_support[iterate][0,0], subregion_support[iterate][0,1]]
        y_coordinates_sub_r_2 = [subregion_support[iterate][1,0], subregion_support[iterate][1,0]]

        x_coordinates_sub_r_3 = [subregion_support[iterate][0,1], subregion_support[iterate][0,1]]
        y_coordinates_sub_r_3 = [subregion_support[iterate][1,0], subregion_support[iterate][1,1]]

        x_coordinates_sub_r_4 = [subregion_support[iterate][0,0], subregion_support[iterate][0,1]]
        y_coordinates_sub_r_4 = [subregion_support[iterate][1,1], subregion_support[iterate][1,1]]

        plt.plot(x_coordinates_sub_r_1, y_coordinates_sub_r_1, listStyle[iterate])
        plt.plot(x_coordinates_sub_r_2, y_coordinates_sub_r_2, listStyle[iterate])
        plt.plot(x_coordinates_sub_r_3, y_coordinates_sub_r_3, listStyle[iterate])
        plt.plot(x_coordinates_sub_r_4, y_coordinates_sub_r_4, listStyle[iterate])
    
    plt.plot(orig_samples[:,0], orig_samples[:,1], ".")
    
    for i, subregionPoints in enumerate(regionSamples):
        if subregionPoints.shape[0]!=0:
            plt.plot(regionSamples[i][:,0], regionSamples[i][:,1], listStyle_marker[i])
    plt.show()
