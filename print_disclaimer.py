class print_disclaimer:
    def print_exception():
        print("Try Again. Invalid input for calculating the slope and y-intercept for Least Squared")
    def print_predicted(self, pred_xy):
        self.pred_xy = pred_xy
        print("====  x & y_value (Predicted using Least Squared) ====")
        print(self.pred_xy)
    def print_predicted_sum(self, sum_y_pred_list):
        self.sum_y_pred_list = sum_y_pred_list
        print("***Since the mapping is based on the minimum sum of least_squared, we can map y-pred to y-ideal value.***") 
        print("==== Each sum of y_value (Predict) ====")
        print(self.sum_y_pred_list)
    def print_ideal_sum(self, sum_y_ideal_list):
        self.sum_y_ideal_list = sum_y_ideal_list
        print("==== Each sum of y_value (Ideal) ====")
        print(sum_y_ideal_list)
    def print_index(self, final_index):
        self.final_index = final_index
        print()
        print("As result, we will derive to these 4 indices which has the minimum least_squared")
        print("Note: Index starts from 0 while ideal y-value starts from y1")
        print(self.final_index)
        print()
    def print_ideal(self, df_final_ideal):
        self.df_final_ideal = df_final_ideal
        print()
        print("From the indexes mapping, here are the 4 ideal functions")
        print(self.df_final_ideal)
    def print_test_data(self, df_query_test):
        self.df_query_test = df_query_test
        print("==== Test Dataset ====")
        print(self.df_query_test)
        print()
    def print_new_y_test(self, df_new_y_test):
        self.df_new_y_test = df_new_y_test
        print("From here, we go through each point, save all data points which are no smaller than Sqrt(2)")
        print("""==== New Test Dataset ====""")
        print(self.df_new_y_test)
    def print_removed(self, removed_df):
        self.removed_df = removed_df
        print("Here is the removed data points and its indices")
        print("""==== Removed Data points ====""")
        print(self.removed_df)
    def print_y_dev(self, y_dev_list):
        self.y_dev_list = y_dev_list
        print("In addition, a list of all y-deivation values are saved as well")
        print(self.y_dev_list)
        print()
    def print_new_y(self, df_new_y_test):
        self.df_new_y_test = df_new_y_test
        print("Therefore, we drop those data points to match the indices with the x values, so the Test Dataset will be: ")
        print("""==== Test Dataset (with Initial indices) ====""")
        print(self.df_new_y_test)
    def print_conclusion(self):
        print("Based on observation from the popped graph, we can conclude that the test data does fits to one or two of the ideal functions picked.")
        print("The functions that the data points touches are y8 and y34")