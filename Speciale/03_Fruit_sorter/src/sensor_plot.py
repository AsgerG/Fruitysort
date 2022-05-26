from cProfile import label
from cgitb import enable
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import lines

test_name = "lol"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_two_close_apples.csv"
csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_two_very_close_not_touching_apples_155747.csv"





df = pd.read_csv(csv_path,index_col=0)

len(df)


df.reset_index().plot.scatter(y='0',x='index',title=("Video sensor: " + test_name))
df.reset_index().plot.scatter(y='1',x='index', title=("Push sensor: " + test_name))

df.plot(y='0', use_index=True)

df.reset_index().plot.scatter(y='0',x='index',title=("Video sensor: " + test_name))


ax = plt.gca()
df.plot(y='0', use_index=True, ax=ax)
df.plot(y='1', use_index=True, ax=ax)
plt.show



ax = plt.gca()

df.reset_index().plot.scatter(y='0',x='index',title=("Video sensor: " + test_name), ax=ax)
df.reset_index().plot.scatter(y='1',x='index', title=("Push sensor: " + test_name), color = "red", ax=ax)

#df2.reset_index().plot(y='cubes',x='index', title=("Push sensor: " + test_name), color = "red", ax=ax)

plt.show





  



#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_two_close_apples.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_two_very_close_not_touching_apples_155747.csv"

#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_2_cm_distance2022-05-20_1210.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_3_cm_distance2022-05-20_1215.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_4_cm_distance_no_push2022-05-20_1221.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_4_cm_distance_was_supposed_to_push_second2022-05-20_1222.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_4_cm_distance2022-05-20_1223.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_5_cm_distance2022-05-20_1226.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_6_cm_distance_pushed_first_but_collision2022-05-20_1229.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_6_cm_distance2022-05-20_1234.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_7_cm_distance_orange2022-05-20_1401.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_7_cm_distance_orange2022-05-20_1403.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_7_cm_distance2022-05-20_1359.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_7_cm_distance2022-05-20_1400.csv"


def plot_this(csv_path):
    df = pd.read_csv(csv_path,index_col=0)

    # Camera sensor
    #ax = plt.gca()
    array = np.zeros(len(df))
    beta_thresh = array + 10
    alpha_thresh = array + 6

    column_values = ['thresh']
    df_beta = pd.DataFrame(data = beta_thresh, columns = column_values)
    df_alpha = pd.DataFrame(data = alpha_thresh, columns = column_values)
    #df_beta.plot(y='thresh', use_index=True, ax=ax)
    #df_alpha.plot(y='thresh', use_index=True, ax=ax)

    plt.figure(figsize = (20,6))
    plt.ylabel = "distance / cm"
    ax = plt.gca()
    df.plot(y='0', use_index=True, ax=ax, label="sensor data", color='dodgerblue')
    #df.plot(y='1', use_index=True, ax=ax)
    df.reset_index().plot.scatter(y='0',x='index', ax=ax,color='dodgerblue')
    #df.reset_index().plot.scatter(y='1',x='index', title=("Push sensor: " + test_name), color = "orange", ax=ax)
    df_beta.reset_index().plot(y='thresh',x='index',linestyle='--', label="motor-stop enabling", color = "green", ax=ax)
    df_alpha.reset_index().plot(y='thresh',x='index',linestyle='--', label="motor-stop", color = "red", ax=ax)
    ax.set_xlabel("measurements index", fontsize=16)
    ax.set_ylabel("distance / cm", fontsize=16)
    ax.set_title("Video Sensor: Two apples", fontsize=20)
    ax.set_ylim([0, 20])
    plt.yticks((np.arange(0,11)*2))
    ax.set_xlim([0, 40])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    plt.legend([lines.Line2D([0], [0], marker='o', ls='-', c='dodgerblue'),
                lines.Line2D([0], [0], ls='--', c='green'),
                lines.Line2D([0], [0], ls='--', c='red'),
                red_patch],
                ["sensor data", "motor stop enabeling     ", "motor stop"], fontsize=16,loc='center left', bbox_to_anchor=(1, 0.5),
                borderpad=1,labelspacing=2)

    plt.show





    # Push sensor
    end_time = 7.535157
    start_time = 0.29007
    num_measurements = 25
    time_per_measurement = (end_time-start_time)/num_measurements
    frequency = 1/time_per_measurement
    array = np.zeros(len(df))
    push_thesh = array+6
    column_values = ['thresh']
    df_push = pd.DataFrame(data = push_thesh, columns = column_values)

    sensor_color = "dodgerblue"
    enabeling_period_color = "limegreen"
    alpha_value= 0.4
    plt.figure(figsize = (20,6))
    ax = plt.gca()
    plot1 = df.plot(y='1', use_index=True, color=sensor_color, ax=ax, label="sensor data")
    df.reset_index().plot.scatter(y='1',x='index', title=("Push sensor: " + test_name), color=sensor_color, ax=ax)
    df_push.reset_index().plot(y='thresh',x='index',linestyle='--', label="motor-stop", color = "red", ax=ax)
    x1 = np.array([21,21+frequency])
    ax.fill_between(x1, 0, 1, color=enabeling_period_color, alpha=alpha_value, transform=ax.get_xaxis_transform())
    x2 = np.array([26,26+frequency])
    ax.fill_between(x2, 0, 1, color=enabeling_period_color, alpha=alpha_value, transform=ax.get_xaxis_transform())
    red_patch = mpatches.Patch(color=enabeling_period_color, label='The red data', alpha=alpha_value)
    ax.set_xlabel("measurements index", fontsize=16)
    ax.set_ylabel("distance / cm", fontsize=16)
    ax.set_title("Push Sensor: Two apples", fontsize=20)
    #plt.legend(fontsize=16, loc='lower right',handles=[plot1, red_patch])
    ax.set_ylim([0, 20])
    ax.set_xlim([0, 40])
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend([lines.Line2D([0], [0], marker='o', ls='-', c=sensor_color), red_patch,
                lines.Line2D([0], [0], ls='--', c='red')],
                ["sensor data", "trigger enabeling period","sorting trigger"], fontsize=16,loc='center left', bbox_to_anchor=(1, 0.5),
                borderpad=1, labelspacing=2)

    plt.yticks((np.arange(0,11)*2))
    plt.show





#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_2_cm_distance2022-05-20_1210.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_3_cm_distance2022-05-20_1215.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_4_cm_distance_no_push2022-05-20_1221.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_4_cm_distance_was_supposed_to_push_second2022-05-20_1222.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_4_cm_distance2022-05-20_1223.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_5_cm_distance2022-05-20_1226.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_6_cm_distance_pushed_first_but_collision2022-05-20_1229.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_6_cm_distance2022-05-20_1234.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_7_cm_distance_orange2022-05-20_1401.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_7_cm_distance_orange2022-05-20_1403.csv"
#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_7_cm_distance2022-05-20_1359.csv"
csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_7_cm_distance2022-05-20_1400.csv"

plot_this(csv_path)




#csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_two_close_apples.csv"
csv_path = "/Users/oas/Desktop/Asger/Fruitysort/Speciale/03_Fruit_sorter/data/data_csv/sensor_test_two_very_close_not_touching_apples_155747.csv"
plot_this(csv_path)
