# sourcery skip: for-index-underscore, remove-unreachable-code, square-identity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from statistics import pvariance
import math

# import os
from pathlib import Path
import traceback
import chardet
# os.system("color")
#from simple_colors import *

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QApplication,
    QLabel,
    QWidget,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QStatusBar,
)
from PyQt5 import uic
from PyQt5.QtGui import QPixmap


class Analyze:
    def __init__(self, parent=None):

        super().__init__()
        self.parent = parent
        self.folder_name = ""
        self.folder_path = ""
        self.name = ""
        self.f_name = ""
        self.county_key = ""
        self.census_id = 0
        self.t_spacing = 0
        self.total_transects = 0
        self.sampled_transects = 0
        self.study_area = 0
        self.Ez2 = 0
        self.Ez_2 = 0
        self.R = 0
        self.s_f = 0
        self.Ez = 0
        

    def set_county_key(self, county_key):
        self.county_key = county_key

    def unit_area(self):
        # directions N-S=1, E_W =2
        script_path = Path(__file__).resolve()
        df = pd.read_excel(
            self.f_name, sheet_name="RSO" + self.census_id, dtype={"PST": str}
        )
        df_fso = pd.read_excel(
            self.f_name,
            sheet_name="FSO" + self.census_id,
            dtype={
                "LENGTH": float,
            },
        )
        df_length = pd.read_csv(Path(script_path.parent, "unit_lengths.csv"))
        units_with_error = pd.read_csv(Path(self.folder_path, "Re_check this units .csv"))
        units_with_error_list = units_with_error["UNIT"].tolist()
        filtered_df_fso = df_fso[~df_fso["UNIT1"].isin(units_with_error_list)]

        # Replace null values with 0'
        # df.fillna(0, inplace=True)
        # Convert non-numeric values to numeric values
        # df = df.apply(pd.to_numeric, errors="coerce")

        # unit_no = list(df_fso["UNIT"])
        unit = np.array(filtered_df_fso["UNIT1"])
        dir_ = np.array(filtered_df_fso["DIR1"])
        unit_radar = np.array(filtered_df_fso["RADAR1"])
        unit_length = np.array(df_length["LENGTH"])

        df_fso["DIR1"] = df_fso["DIR1"].astype(str)
        df_length["DIRECTION"] = df_length["DIRECTION"].astype(str)

        # Replace values in the "DIR1" column
        filtered_df_fso = filtered_df_fso.copy()
        filtered_df_fso.loc[:, "DIR1"] = filtered_df_fso["DIR1"].replace({"N": "1", "S": "1", "W": "2", "E": "2"})
        filtered_df_fso.loc[:, "County"] = self.county_key


        # Join the dataframes using "dir_" and "unit" columns
        merged_df = pd.merge(
           filtered_df_fso,
            df_length,
            left_on=["DIR1", "UNIT1", "County"],
            right_on=["DIRECTION", "UNIT", "COUNTY"],
            how="left",
        )

        # Filter rows with null values in LENGTH column
        filtered_df = merged_df[merged_df["LENGTH"].isnull()]

        # Filter rows with non-null values in LENGTH column
        merged_df = merged_df[merged_df["LENGTH"].notnull()]

        file_path_no_lengths = Path(self.folder_path, "Error_units with no lengths.csv")
        file_path_length = Path(self.folder_path1, "FSO_Data_With_Lenths.xlsx")

        filtered_df.to_csv(file_path_no_lengths, index=False)
        merged_df.to_excel(file_path_length, index=False)

        # sample study area
        # Create a list of dictionaries where each dictionary represents
        # the unit area for each row

        df_areacalc = pd.read_excel(file_path_length)

        unit = np.array(df_areacalc["UNIT1"])
        dir_ = np.array(df_areacalc["DIR1"])
        unit_radar = np.array(df_areacalc["RADAR1"])
        unit_length = np.array(df_areacalc["LENGTH"])

        h = 400  # nominal fying height
        w = 282  # nominal strip width using the said fying height

        unit_area_list = []
        total_sample_area = 0  # initialize the sumto zero
        error_message = ""  # Initialize error_message with an empty string
        for i in range(len(unit_radar)):
            try:
                a = unit_radar[i]
                f = float(a)
                # ratio = Target strip width(m)/Target flying height(m)=1.1562
                unit1 = unit[i]  # Get the 'unit' value from the input data
                if pd.isnull(unit_length[i]) == False:
                    try:
                        length = np.divide(float(unit_length[i]), 1000)
                        actual_strip_width = (w * ((f * 10) / 400)) / 1000
                        unit_area = actual_strip_width * length

                        total_sample_area += unit_area
                        unit_area_list.append(
                            {"Unit": unit1, "Unit Area(km^2)": unit_area}
                        )
                    except ValueError:
                        error_massage = (
                            "Error: Non-numeric value found in unit_length array."
                        )
                       # QMessageBox.warning(self.parent, "Warning", error_message1)
            except ValueError:
                error_message = f"Error: Non-numeric value found in radar column in FSO data array at index {i}. Ignoring."
                #QMessageBox.warning(self.parent, "Warning", error_message)
                continue
            # Display all error messages at once
        if error_message:
            QMessageBox.warning(
            self.parent,"Warning"," ".join(error_message))
        
        self.Ez = total_sample_area
        # Create a pandas DataFrame from the list of dictionaries
        df = pd.DataFrame(unit_area_list)
        # Add a new column that squares each unit area (ha)
        df["Unit Area Squared(km^2)"] = df["Unit Area(km^2)"].apply(lambda x: x**2)

        # Calculate the sum of squared unit area (ha^2)
        sum_squared_unit_area = df["Unit Area Squared(km^2)"].sum()
        self.Ez2 = sum_squared_unit_area
        self.Ez_2 = self.Ez * self.Ez

        # print(yellow("Total area sampled is"), blue(Ez))
        # print(Ez_2)
        # print(Ez2)
        # Save the DataFrame to an excel file called 'unit_areas.xlsx'
        file_path = Path(self.folder_path1, "unit_areas_km_squared.xlsx")
        df.to_excel(file_path, index=False)

        # df.to_excel(self.folder_path1 + "/unit_areas_ha.xlsx", index=False)

        df = pd.read_excel(
            self.f_name, sheet_name="RSO" + self.census_id, dtype={"PST": str}
        )

        # Create a new dataframe to store the results
        result_df = pd.DataFrame(columns=["Column", "Sum", "Variance"])

        # enter CENSUS ID
        c = self.census_id
        # Enter Transects Spacing
        t = self.t_spacing

        # enter number of transects in the population
        N = self.total_transects

        # enter number of transects sampled
        n = self.sampled_transects

        # enter total study area
        z = self.study_area

        # enter total area sampled
        # Ez = float(input("Enter total area sampled (km sq)(Ez):  \n"))
        # print(z)

        # Iterate over each column in the dataframe except the first 4 columns)
        results = []
        zero_count = []
        self.s_f = np.round(float(self.Ez / z) * 100, 3)

    def zerro_count(self):
        results = []
        zero_count = []
        zero_count_df = pd.DataFrame(columns=["species (Zero count)"])
        df = pd.read_excel(
            self.f_name, sheet_name="RSO" + self.census_id, dtype={"PST": str}
        )
        for column in df.columns[4:]:
            # Convert column values to numeric type
            numeric_values = pd.to_numeric(df[column], errors="coerce").fillna(0)
            # Compute the sum of each column
            # column_sums = df[column].sum()
            column_sums = numeric_values.sum()
            # Identify non-numeric values
            non_numeric_values = df[column][pd.isna(numeric_values)]

            exclude_columns = [
                "CX",
                "MH",
                "MS",
                "MP",
                "MT",
                "TH",
                "TS",
                "TS",
                "SR",
                "FD",
            ]
            if (
                column_sums == 0 and column not in exclude_columns
            ):  # Exclude these column
                zero_count.append({"species (Zero count)": column})
                zero_count_df = pd.DataFrame(
                    zero_count, columns=["species (Zero count)"]
                )
            # Print non-numeric values
            if not non_numeric_values.empty:
                print(f"Column '{column}' contains non-numeric values")

        file_path1 = Path(self.folder_path, "SPECIES WITH ZERO SAMPLE COUNT.csv")
        zero_count_df.to_csv(file_path1, index=False)
        # return zero_count_df

        # print('\033[1m'+yellow('Check output of units with zero count and with other errors'))

        df = pd.read_excel(
            self.f_name, sheet_name="RSO" + self.census_id, dtype={"PST": str}
        )
        pst = np.array(df["PST"])

        unit_no = list(df["UNIT"])
        fl_no = list(df["FLIGHT"])
        count = []
        count1 = []
        f = []
        k = []
        unit = []

        # Group the DataFrame by the two columns and filter based on the count
        filtered_df = df.groupby("UNIT").filter(lambda x: len(x) != 2)

        # Calculate the count of each unique combination
        counts = (
            filtered_df.groupby("UNIT")
            .size()
            .reset_index(name="Unit_Count")
        )

        # Merge the counts with the filtered DataFrame
        filtered_df = filtered_df.merge(counts, on="UNIT", how="left")

        # Move the 'Count' column after the 'UNIT' column
        filtered_df.insert(
            filtered_df.columns.get_loc("UNIT") + 1,
            "Unit_Count",
            filtered_df.pop("Unit_Count"),
        )

        # Save the filtered DataFrame with the count column to an Excel file
        filtered_df.to_csv(
            Path(self.folder_path, "Re_check this units .csv"), index=None
        )
           
    def sum_y_squared(self):
        # Compute the sum of each column and square the result
        sum_squared = pd.DataFrame(columns=["SPECIES", "SUM", "SUM_SQUARED"])
        df = pd.read_excel(
            self.f_name, sheet_name="RSO" + self.census_id, dtype={"PST": str}
        )
        for column in df.columns[4:]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            #filtered_df = df[df["UNIT"].duplicated(keep=False)]
            filtered_df = df.groupby("UNIT").filter(lambda x: len(x) == 2)
            column_sums = filtered_df[column].sum()
            Ey = column_sums
            column_sum_sqrd = filtered_df[column].sum() ** 2
            sum_squared["SUM_SQUARED"] = pd.to_numeric(
                sum_squared["SUM_SQUARED"], errors="coerce"
            )
            # column_sum_sqrd = column_sum_sqrd.reset_index()
            new_row = {
                "SPECIES": column,
                "SUM": column_sums,
                "SUM_SQUARED": column_sum_sqrd,
            }
            sum_squared = pd.concat(
                [sum_squared, pd.DataFrame([new_row])], ignore_index=True
            )
            file_path3 = Path(self.folder_path1, "summed y squared.xlsx")
            sum_squared.to_excel(
                file_path3, index=False, header=["SPECIES", "SUM", "SUM_SQUARED"]
            )

            # calculating unit squared summed
            grouped_df = df.groupby("UNIT").filter(lambda x: len(x) == 2)
            grouped_df = grouped_df.groupby("UNIT").sum(numeric_only=True)
            # print(grouped_df)
            grouped_df = (grouped_df) ** 2
            # Sum the squares of each column
            column_sum = grouped_df.sum()
            column_sum = column_sum.reset_index()

            # Write the column sums to a new Excel file
            file_path3a = Path(self.folder_path1, "unit squared summed.xlsx")
            column_sum.to_excel(
                file_path3a, index=False, header=["SPECIES", "unit squared summed"]
            )

            df1 = pd.read_excel(file_path3a)
            s = np.array(df1["SPECIES"])
            # sq=np.array(df1['SUM_SQUARED'])
            # SP=list(df1['SPECIES'])
            for column in df1.columns[1:]:
                df1["unit squared summed"] = pd.to_numeric(
                    df1["unit squared summed"], errors="coerce"
                )
                Ey_sqrd = df1["unit squared summed"]

            # merging unit squared and summed y squared excels
            file_path3 = Path(self.folder_path1, "summed y squared.xlsx")
            df2 = pd.read_excel(file_path3)
            filtered_df = df2[df2["SUM"] != 0]
            merge_df = pd.merge(
                filtered_df,
                df1[["SPECIES", "unit squared summed"]],
                on="SPECIES",
                how="left",
            )
            file_path4 = Path(self.folder_path1, "combined.xlsx")
            merge_df.to_excel(file_path4, index=False)

            # analysis

    def analysis(self):
        script_path = Path(__file__).resolve()
        df_names = pd.read_csv(
            Path(
                script_path.parent,
                "Abbreviations and names of animals and structures.csv",
            )
        )
        file_path4 = Path(self.folder_path1, "combined.xlsx")
        df3b = pd.read_excel(file_path4)
        # cattle all sumation
        # Specify the row names you want to sum
        cattle_rows = ["CF", "CP", "CL", "CP"]

        # Sum the specified rows and store the result in a new row
        selected_row_sum = df3b[df3b["SPECIES"].isin(cattle_rows)].sum(skipna=True)

        # Specify the name for the new row
        new_row_name = "CX"

        # Create a new row with the summed output
        new_row = pd.DataFrame([selected_row_sum], columns=df3b.columns)
        new_row["SPECIES"] = "CX"

        # Append the new row to the DataFrame
        df3a = pd.concat([df3b, new_row], ignore_index=True)
        # Save the modified DataFrame back to the Excel file:
        file_path4b2 = Path(self.folder_path1, "combined.xlsx")
        # df3a.to_excel(file_path4, index=False)

        merge_df2 = pd.merge(
            df3a,
            df_names[["SPECIES", "SPECIES NAME"]],
            on="SPECIES",
            how="left",
        )
        file_path4b = Path(self.folder_path1, "combined_with names.xlsx")
        merge_df2.to_excel(file_path4b, index=False)
        df3 = pd.read_excel(file_path4b)

        SP = np.array(df3["SPECIES NAME"])
        ABV = np.array(df3["SPECIES"])
        column_sums = np.array(df3["SUM"])
        Eyallsqrd = np.array(df3["SUM_SQUARED"])
        Ey_sqrd = np.array(df3["unit squared summed"])

        # average species density factor-overall density
        self.R = column_sums / self.Ez  # sample mean
        Ey = column_sums
        Ey_sum = np.sum(Ey)
        # Eyallsqrd=column_sum_sqrd
        Ezy_i = Ey * self.Ez
        Ezy = np.sum(Ezy_i)
        self.n = float(self.total_transects)
        Sy2 = np.round(
            (1 / (self.sampled_transects - 1))
            * ((Ey_sqrd) - ((Eyallsqrd) / self.sampled_transects)),
            3,
        )
        # print(Sy2)
        # Sy2 = 94878411.0203252
        # count based covariance
        Sz2 = np.round(
            (1 / self.sampled_transects - 1)
            * (self.Ez2 - (self.Ez_2 / self.sampled_transects)),
            3,
        )
        # area count covariance
        Szy = np.round(
            (1 / (self.sampled_transects - 1))
            * ((Ezy) - ((self.Ez * Ey_sum) / self.sampled_transects)),
            3,
        )
        # pop estimate
        Y = np.round((self.study_area * self.R), 0)
        # pop variance
        column_variances = np.round(
            np.abs(
                (
                    (
                        self.total_transects
                        * (self.total_transects - self.sampled_transects)
                    )
                    / self.sampled_transects
                )
                * Sy2
            ),
            0,
        )
        # -(2*R*Szy)) + ((R*R) * Sz2),2)
        # Compute the standard error of each column
        std_err = np.round(np.sqrt(column_variances), 3)
        # Compute the sum of each column
        df_output = pd.DataFrame(
            {
                "SPECIES": SP,
                "Abbreviations": ABV,
                "Sample sum": column_sums,
                "Population_Estimate": Y,
                "Population_Variance": column_variances,
                "Standard Error": std_err,
                "CENSUS ID": self.census_id,
                "TOTAL AREA sq km": self.study_area,
                "AREA SAMPLED sq km": self.Ez,
                "sampling Fraction %": self.s_f,
                "Transect spacing (KM)": self.t_spacing,
            }
        )

        df_output.loc[
            1:,
            [
                "CENSUS ID",
                "TOTAL AREA sq km",
                "AREA SAMPLED sq km",
                "sampling Fraction %",
                "Transect spacing (KM)",
            ],
        ] = np.nan

        # file_name2 = self.folder_name + "_statistics.txt"
        file_name = self.name + "_statistics.xlsx"

        file_name_structures = self.name + "_structures.xlsx"

        # animal statistics file path
        file_path_f = Path(self.folder_path, file_name)

        # structure excel path
        file_path_str = Path(self.folder_path, file_name_structures)

        # df_output.to_csv(file_path_f2, index=False)
        df_output.to_excel(file_path_f, index=False)

        df_stat = pd.read_excel(file_path_f)

        # Define the specific names of structures you want to filter out
        structures = ["MH", "MS", "MP", "MT", "TH", "TS", "TS", "SR", "FD"]

        # structure dataframe
        df_structures = df_stat[df_stat["Abbreviations"].isin(structures)]
        # Select specific columns from df_structures
        df_structures_selected = df_structures[
            ["SPECIES", "Abbreviations", "Sample sum"]
        ]

        # Filter the DataFrame based on the "Abbreviations" column
        df_species = df_stat[~df_stat["Abbreviations"].isin(structures)]

        # save structures
        df_structures_selected.to_excel(file_path_str, index=False)

        # Save the species DataFrame to a new Excel file
        df_species.to_excel(file_path_f, index=False)

    
       

    
class UI(QMainWindow):
    
    cancel_clicked = pyqtSignal()
    

    def __init__(self):
        super(UI, self).__init__()
        # load the ui file
        script_path = Path(__file__).resolve()
        ui_path = Path(script_path.parent, "view.ui")
        uic.loadUi(ui_path, self)
        self.analyzer = Analyze()  # Create an instance of the analyze class
        
        # Set the background image using stylesheet
        self.setStyleSheet(
            "background-image: url(Path(script_path.parent, 'ANIMALS.png'); background-repeat: no-repeat; background-position: center;"
        )
        # define widgets
        self.label_1 = self.findChild(QLabel, "label_1")
        self.label_2 = self.findChild(QLabel, "label_2")
        self.label_3 = self.findChild(QLabel, "label_3")
        self.label_4 = self.findChild(QLabel, "label_4")
        self.label_5 = self.findChild(QLabel, "label_5")
        self.label_6 = self.findChild(QLabel, "label_6")
        self.label_7 = self.findChild(QLabel, "label_7")
        self.label_8 = self.findChild(QLabel, "label_8")
        self.label_9 = self.findChild(QLabel, "label_9")

        self.lineEdit_1 = self.findChild(QLineEdit, "lineEdit_1")
        self.lineEdit_2 = self.findChild(QLineEdit, "lineEdit_2")
        self.lineEdit_3 = self.findChild(QLineEdit, "lineEdit_3")
        self.lineEdit_4 = self.findChild(QLineEdit, "lineEdit_4")
        self.lineEdit_5 = self.findChild(QLineEdit, "lineEdit_5")
        self.lineEdit_6 = self.findChild(QLineEdit, "lineEdit_6")
        self.lineEdit_7 = self.findChild(QLineEdit, "lineEdit_7")
        self.lineEdit_8 = self.findChild(QLineEdit, "lineEdit_8")

        self.pushbutton_1 = self.findChild(QPushButton, "pushButton_1")
        self.pushbutton_2 = self.findChild(QPushButton, "pushButton_2")
        self.pushbutton_3 = self.findChild(QPushButton, "pushButton_3")
        self.pushbutton_4 = self.findChild(QPushButton, "pushButton_4")
        self.pushbutton_5 = self.findChild(QPushButton, "pushButton_5")
        self.pushbutton_6 = self.findChild(QPushButton, "pushButton_6")

        # Set float validator for lineEdits
        float_validator = QDoubleValidator()
        # float_validator.setDecimals(2)  # Optional: Set the number of decimal places

        self.lineEdit_3.setValidator(float_validator)
        self.lineEdit_4.setValidator(float_validator)
        self.lineEdit_5.setValidator(float_validator)
        self.lineEdit_6.setValidator(float_validator)

        # Set integer validator for lineEdit_3
        int_validator = QIntValidator()
        self.lineEdit_2.setValidator(int_validator)

        self.pushbutton_1.clicked.connect(self.input)
        self.pushbutton_2.clicked.connect(self.output)
        self.pushbutton_3.clicked.connect(self.runprogram)
        self.pushbutton_4.clicked.connect(self.cancel)
        self.pushbutton_5.clicked.connect(self.clear_all_line_edits)
        self.pushbutton_6.clicked.connect(self.graph)


        # self.lineEdit_1.textChanged.connect(self.getLineEditText)
        self.lineEdit_1.editingFinished.connect(self.countyname)
        self.lineEdit_2.editingFinished.connect(self.census_ID)
        self.lineEdit_3.editingFinished.connect(self.transectspacing)
        self.lineEdit_4.editingFinished.connect(self.totaltransects)
        self.lineEdit_5.editingFinished.connect(self.sampletransects)
        self.lineEdit_6.editingFinished.connect(self.totalstudyarea)

        # show the app
        self.show()
        # instance variable to store the selected folder
        self.out_folder = ""
        # Initialize folder_path1 and folder path as an empty string
        self.folder_path1 = ""
        self.folder_path = ""
        self.folder_name = ""
        self.name = ""
        self.county_key = ""
        self.file_path_f = ""
    

    def cancel(self):
        self.close()
        self.cancel_clicked.emit()

    def clear_all_line_edits(self):
        self.lineEdit_1.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.lineEdit_4.clear()
        self.lineEdit_5.clear()
        self.lineEdit_6.clear()

        self.lineEdit_7.setReadOnly(False)
        self.lineEdit_8.setReadOnly(False)

        self.lineEdit_7.clear()
        self.lineEdit_8.clear()
    
    def graph(self):
        analyzer = Analyze(self)
        analyzer.folder_path1 = self.folder_path1
        analyzer.folder_path = self.folder_path
        analyzer.name = self.name

        file_name = self.name + "_statistics.xlsx"
        file_path_f = Path(self.folder_path, file_name)

        df_g = pd.read_excel(file_path_f)
       
        x_values = df_g['SPECIES']
        y_values = df_g['Population_Estimate']
        y_values_log = np.log(y_values)
        y_values_log = np.round(y_values_log,1).astype(int)
        # Create separate figures and axes for the line plot and bar graph
        '''

        # Figure 1: Line plot
        plt.figure()
        plt.plot(x_values, y_values)
        plt.xlabel('SPECIES')
        plt.ylabel('Population_Estimate')
        plt.title('Species Population Estimates Line Plot (from collected samples)')
        plt.xticks(rotation=90)
        # Display Figure 1
        plt.show()
        '''
        # Figure 2: Bar graph
        plt.figure(figsize=(16,10))
        bars = plt.bar(x_values, y_values_log)
        plt.xlabel('Animal Species')
        plt.ylabel('Population_Estimate (Log)')
        title = (self.name +' Species Population Estimates (Log) Bar Graph (from collected samples)')
        plt.title(title.title())
        plt.xticks(rotation=90)
        # Add labels to the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='top', rotation=90)

        # Set logarithmic scale on the y-axis
        plt.yscale('log')
        # Display both figures separately
        plt.tight_layout()
        plt.show()
       
    def runprogram(self):
        # Create a progress bar
        progress_bar = QProgressBar(self)
        progress_bar.setGeometry(100, 84, 400, 25)
        progress_bar.setValue(10)
        progress_bar.show()

        try:
            self.createfolder()
            analyzer = Analyze(self)
            analyzer.name = self.lineEdit_1
            analyzer.f_name = (
                self.f_name
            )  # Set the f_name attribute with a valid file path
            analyzer.census_id = (
                self.lineEdit_2.text()
            )  # Set the census_id attribute of the analyze instance
           
            analyzer.t_spacing = self.handleFloatLineEdit(self.lineEdit_3)
            analyzer.total_transects = self.handleFloatLineEdit(self.lineEdit_4)
            analyzer.sampled_transects = self.handleFloatLineEdit(self.lineEdit_5)
            analyzer.study_area = self.handleFloatLineEdit(self.lineEdit_6)

            analyzer.folder_path1 = self.folder_path1
            analyzer.folder_path = self.folder_path
            analyzer.name = self.name
            analyzer.set_county_key(self.county_key)  # Set the county_key value

            error_message = f"Name: {analyzer.name}\nCounty Key: {analyzer.county_key}\n"  # Include analyzer.name and analyzer.county_key in the error message
            # Update the progress bar value as the program progresses
            progress_bar.setValue(25)  # Update with appropriate value (25%)
            analyzer.zerro_count()
            analyzer.unit_area()
             # Update the progress bar value as the program progresses
            progress_bar.setValue(50)
            analyzer.sum_y_squared()
            
            analyzer.analysis()

            # Update the progress bar again
            progress_bar.setValue(100)  # Update to indicate completion

            # Close the progress bar after the program execution is complete
            

            # Create the pop-up message box
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Analysis Completed")
            msg_box.setText("Analysis completed!")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()

        except Exception as e:
            error_message += (
                f"An error occurred:\n{str(e)}\nSee Terminal for this specific message"
            )
            print(error_message)
            traceback.print_exc()  # Print the traceback for more detailed error information # Print the traceback for more detailed error information

            # Create the pop-up message box for error
            error_box = QMessageBox()
            error_box.setWindowTitle("Error")
            error_box.setText(error_message)
            error_box.setIcon(QMessageBox.Critical)
            error_box.exec_()
        progress_bar.close()
    def output(self):
        out_folder = Path(
            QFileDialog.getExistingDirectory(self, "Select output Directory")
        )

        if out_folder:
            self.out_folder = out_folder
            self.lineEdit_8.setText(str(out_folder))
            self.lineEdit_8.setReadOnly(True)

    def countyname(self):
        self.name = self.lineEdit_1.text()
        words = self.name.split()  # Extract the first four words
        self.county_key = " ".join(words).lower()[
            :4
        ]  # Join the extracted words back into a string and convert to lowercase

    def census_ID(self):
        census_id = self.lineEdit_2.text()
        int_value, is_int = self.tryParseInt(census_id)
        if is_int:
            census_id = str(int_value)
        else:
            self.lineEdit_2.clear()

    def tryParseInt(self, text):
        try:
            int_value = int(text)
            return int_value, True
        except ValueError:
            return 0, False

        # floats

    def tryParseFloat(self, text):
        try:
            float_value = float(text)
            return float_value, True
        except ValueError:
            # Check if the input is an integer without a decimal point
            if text.isdigit():
                return float(text), True  # Treat it as a float
            else:
                return 0.0, False

    def handleFloatLineEdit(self, lineEdit):
        input_text = lineEdit.text()
        float_value, is_float = self.tryParseFloat(input_text)
        if is_float:
            lineEdit.setText(str(float_value))
            return float_value  # Return the float value
        else:
            lineEdit.clear()
            return 0.0  # Or any other suitable default value

    def transectspacing(self):
        t_spacing = self.handleFloatLineEdit(self.lineEdit_3)

    def totaltransects(self):
        total_transects = self.handleFloatLineEdit(self.lineEdit_4)

    def sampletransects(self):
        sampled_transects = self.handleFloatLineEdit(self.lineEdit_5)

    def totalstudyarea(self):
        study_area = self.handleFloatLineEdit(self.lineEdit_6)

        # print(float_4)

    def createfolder(self):
        # self.lineEdit_8.setText(out_folder)
        parent_dir = self.out_folder

        name = self.lineEdit_1.text()
        # Define the name of the folder you want to create
        self.folder_name = name
        # print( folder_name)
        folder_name2 = "program files _ignore"
        # Create a folder if it doesn't exist
        # self.folder_path = os.path.join(parent_dir, name)
        self.folder_path = Path(parent_dir, name)
        if not self.folder_path.exists():
            self.folder_path.mkdir()
            # Set self.folder_path1 as an instance variable
            self.folder_path1 = Path(parent_dir, name, folder_name2)
            if not self.folder_path1.exists():
                self.folder_path1.mkdir()

    def input(self):
        # Check if all line edits are filled
        if (
            self.lineEdit_2.text() == ""
            or self.lineEdit_3.text() == ""
            or self.lineEdit_4.text() == ""
            or self.lineEdit_5.text() == ""
            or self.lineEdit_6.text() == ""
        ):
            error_message = "Please fill all the required fields."
            QMessageBox.critical(self, "Error", error_message)
            return
        # open file Dialog
        self.f_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "(*.xlsx)")
        # output filename
        if self.f_name:
            self._extracted_from_input_17()

    # TODO Rename this here and in `input`
    def _extracted_from_input_17(self):
        # Display a message box indicating that the program is running
        running_message = "loading input data. Please wait..."
        running_box = QMessageBox()
        running_box.setWindowTitle("Loading Data")
        running_box.setText(running_message)
        running_box.setIcon(QMessageBox.Information)
        running_box.show()
        
        # Process pending events to ensure the running box is displayed
        QApplication.processEvents()

        self.lineEdit_7.setText(self.f_name)

        census_id = self.lineEdit_2.text()  # Retrieve text from lineEdit_2
        t_spacing = self.handleFloatLineEdit(self.lineEdit_3)
        total_transects = self.handleFloatLineEdit(self.lineEdit_4)
        sampled_transects = self.handleFloatLineEdit(self.lineEdit_5)
        study_area = self.handleFloatLineEdit(self.lineEdit_6)
        try:
            with open(self.f_name, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                #print(encoding)
              
            df = pd.read_excel(
                self.f_name, sheet_name="RSO" + census_id, dtype={"PST": str})
            # Reading FSO DATA
            df_fso = pd.read_excel(
                self.f_name,
                sheet_name="FSO" + census_id,
                dtype={
                    "LENGTH": float})
               
        except Exception:
            QMessageBox.critical(self, "Error", "An error occurred\n check Census id")
            self.lineEdit_2.clear()
            self.lineEdit_7.clear()
            running_box.reject()  # Close the message box in case of an error
        # Lock line edits 7 and 8
        self.lineEdit_7.setReadOnly(True)
        running_box.accept()  # Close the message box after successful loading
     
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui_window = UI()
    ui_window.show()
    app.exec_()
