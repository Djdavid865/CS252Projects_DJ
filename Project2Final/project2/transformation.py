'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
David Jin
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        analysis.Analysis.__init__(self, data=data)
        self.orig_dataset = orig_dataset


    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        new_data = self.orig_dataset.select_data(headers)
        new_header2col = {}
        index = 0
        for item in headers:
            new_header2col[item] = index
            index += 1
        
        self.data = data.Data(headers=headers, data=new_data, header2col=new_header2col)

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        coordinates = np.ones([self.data.get_num_samples(), 1])

        copy = self.data.get_all_data()

        return np.hstack((copy, coordinates))

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        tmatrix = np.eye(self.data.get_num_dims()+1)
        for item in range(len(magnitudes)):
            tmatrix[item, -1] = magnitudes[item]

        return tmatrix

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        smatrix = np.eye(self.data.get_num_dims()+1)
        for item in range(len(magnitudes)):
            smatrix[item,item] = magnitudes[item]

        smatrix[-1,-1] = 1

        return smatrix

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        homogeneous = self.get_data_homogeneous()

        translated = self.translation_matrix(magnitudes)

        transposed = homogeneous.T
 
        translated_data = translated@(transposed)
        translated_data = translated_data[:-1,:].T

        new_data = data.Data(headers=self.data.get_headers(),data=translated_data,header2col=self.data.get_mappings())

        self.data = new_data

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        homogeneous = self.get_data_homogeneous()

        scaled = self.scale_matrix(magnitudes)
        

        scaled_data = scaled@homogeneous.T
        scaled_data = scaled_data[:-1,:].T

        new_data = data.Data(headers=self.data.get_headers(),data=scaled_data,header2col=self.data.get_mappings())

        self.data = new_data

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        homogeneous = self.get_data_homogeneous()
        
        transformed_data = C@homogeneous.T
        transformed_data = transformed_data[:-1,:].T

        new_data = data.Data(headers=self.data.get_headers(),data=transformed_data, header2col=self.data.get_mappings())

        self.data = new_data

        return transformed_data

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        min = self.data.data.min()
        min_array = np.empty(self.data.get_num_dims())
        min_array.fill(-1*min)

        range = self.data.data.max()-min
        range_array = np.empty(self.data.get_num_dims())
        range_array.fill(1/range)

        homogeneous = self.get_data_homogeneous()

        translation_matrix = self.translation_matrix(min_array)
        smatrix = self.scale_matrix(range_array)

        normalized = smatrix@translation_matrix@homogeneous.T
        self.data.data = normalized[:-1,:].T

        return normalized

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        min_list = self.min(self.data.get_headers())

        max_list = self.max(self.data.get_headers())
        range = max_list - min_list

        self.translate(-1*min_list)
        self.scale(1/range)

        return self.data.data

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        rotmatrix = np.eye(4)

        axis = self.data.get_mappings()[header]

        rad = np.deg2rad(degrees)
        
        if axis == 0:
            rotmatrix[1,1] = np.cos(rad)
            rotmatrix[1,2] = -np.sin(rad)
            rotmatrix[2,1] = np.sin(rad)
            rotmatrix[2,2] = np.cos(rad)
        elif axis == 1:
            rotmatrix[0,0] = np.cos(rad)
            rotmatrix[0,2] = np.sin(rad)
            rotmatrix[2,0] = -np.sin(rad)
            rotmatrix[2,2] = np.cos(rad)
        elif axis == 2:
            rotmatrix[0,0] = np.cos(rad)
            rotmatrix[0,1] = -np.sin(rad)
            rotmatrix[1,0] = np.sin(rad)
            rotmatrix[1,1] = np.cos(rad)
        else:
            print('Variable')
            return

        return rotmatrix

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        rotate = self.rotation_matrix_3d(header, degrees)
        new_data = self.transform(rotate)[:, :-1]
        self.data = data.Data(headers=self.data.get_headers(), data=new_data, header2col=self.data.get_mappings())

        return new_data

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        colormap = palettable.colorbrewer.sequential.Blues_8
        x_data = self.orig_dataset.select_data([ind_var])
        y_data = self.orig_dataset.select_data([dep_var])
        z_data = self.orig_dataset.select_data([c_var])
        plt.scatter(x_data, y_data, c=z_data, s=75, cmap=colormap.mpl_colormap, edgecolor='white')
        if title != None:
            plt.title("No Title")
