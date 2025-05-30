
#import sys
#print('运行版本是：',sys.version)#检查运行版本是否是python38,如果不是，则修改环境变量

"""
#智能分析你实际用了哪些库
#pip install pipreqs
#cmd进入python文件所在的目录d:/pycharm/project/shinylive
#pipreqs . --force
#生成当前python版本的所有库列表：pip freeze > requirements.txt

1本地运行shiny，检查是否正常
shiny run --reload --launch-browser d:/pycharm/project/shinylive/shinylive20250530.py
2、执行：生成requirements.py。生成某个py文件的包含库
3、注册shinyapps
http://brickwan.shinyapps.io/
python313或python38,运行 c:/python/python313
4、获取token
https://www.shinyapps.io/admin/#/dashboard
5、cmd命令以下两句：
指定python38为临时环境变量（python313使用rsconnect提交时会失败）
set PATH=C:\Python\Python38\;%PATH%

5、连接服务器，填入token。会在shinyapp.io建立一个applications
google账号注册的：
rsconnect add --account brickwan --name brickwan  --token 75999CDA0DE843C59CB7C46B843B1B81 --secret ysKnaCRdmmwtdXT+ls/eRk8ySQgjBHROxA+SGtcy
rsconnect add --account brick-wan --name brick-wan  --token 5C2E455877F881C40F747AF6C4C67B7E --secret kim1P/tz1z3+JTdhn4oaK6xOhslNvTP4D77Kz8L7
#推送代码 指定从shinylive20250530.py运行。默认是app.py
hotmail账号注册的：
rsconnect deploy shiny d:/pycharm/project/shinylive --name brickwan --title brick --entrypoint shinylive20250530.py


测试网址：
http://brickwan.shinyapps.io/

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from shiny import App, ui, reactive, render, Inputs, Outputs, Session
from shiny.types import FileInfo
import io
#import base64
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
#import shap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
import shap
# file_path = Path(__file__).parent / "test_model.csv"
# df_scores = pd.read_csv(file_path)  # ,nrows=96
# X=pd.DataFrame([{'a': 1, 'b': 2,'c': 2,'d': 2,'e': 2,'f': 2,'g': 2,'h': 2,'i': 2}])
# print('X=',X)
# exit()

# 设置matplotlib正常显示中文和负号
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimSun', 'Microsoft YaHei', 'STSong']  # 按照平台设置优先级
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题

# 假设你有0个特征输入框
NUM_FEATURES = 0

# # 自定义 CSS 样式
# custom_css = """
# .plot_container {
#     height: 800px; /* 设置你想要的高度 */
#     border: 1px solid #ccc; /* 可选：添加边框以便于观察 */
# }
# """

# 定义 UI
app_ui = ui.page_fluid(




    ui.row(
        ui.column(

            3,  # 侧边栏宽度

ui.navset_card_tab(
ui.nav_panel("单项预测",
ui.div(
             ui.input_action_button("generate", "预测（测试集：单个）",class_="btn-primary"),

             ui.input_numeric('act_y', '实际目标值', 0,min=0,max=100),#,update_on='blur' placeholder='输入预测值'
            #ui.input_text('pre_y', '预测目标值', '0'),
             ui.tags.p('预测目标值:'),
             ui.output_text_verbatim("result_y_red_text",placeholder=True),

             ui.card(
                 ui.card_header("指标"),
            ui.output_ui("dynamic_inputs_placeholder")
             )

             )
             ),

ui.nav_panel("测试集（内置）",
ui.div(
ui.input_action_button("generate_mul", "预测（内置测试集：全部）" ,class_="btn-info"),
) ),
ui.nav_panel("上传测试集",
ui.div(
ui.input_file("upload_file", "Choose csv File", accept=[".csv"], multiple=False),#.csv,.xlsx
            ui.input_checkbox_group(
                "stats",
                "Summary Stats",
                choices=["Row Count", "Column Count", "Column Names"],
                selected=["Row Count", "Column Count", "Column Names"],
            ),
ui.input_action_button("generate_upload", "开始预测",class_="btn-primary"),
            ui.output_table("summary")
)
             )

)



            #ui.output_text_verbatim("button_status")  # 可选：用于调试，显示按钮点击状态
        ),
        ui.column(9,  # 主内容区域宽度


ui.navset_card_tab(
ui.nav_panel("基础绘图",

ui.div({"style": "width: 100%;"},
ui.output_text_verbatim("result_text"),
ui.output_plot("scatter_plot",height='1000px')#,height='2000px'
)   ),
ui.nav_panel("SHAP绘图",
ui.div({"style": "width: 100%;"},
ui.output_ui("scatter_plot_shap_text")
             ,ui.output_plot("scatter_plot_shap1")
             , ui.output_plot("scatter_plot_shap2")
             ,ui.output_plot("scatter_plot_shap3")
,ui.output_plot("scatter_plot_shap4")
             ) ),


ui.nav_panel("预测绘图",
             ui.div({"style": "width: 100%;"},
ui.output_plot("scatter_plot_pre",height='1000px')#,height='2000px'
             )  )


)
#     ui.tags.div(
# {"style": "height: 900px;"}
# ,ui.output_text_verbatim("result_text"),  #
# ui.output_plot("scatter_plot22",height='2000px'),class_="plot_container")  # 使用 ui.output_plot 直接指定输出类型为 plot


   )



# ui.navset_card_tab(
# ui.nav_panel("基础数据",
#
#              ),
# ui.nav_panel("预测绘图",
# ui.output_text_verbatim("result_text")
#              )
# )


    )
)
#generated_data=''
df= pd.DataFrame({})#全部变量
X=pd.DataFrame({})#
y=''
y_pred=''
#shape=''#
shap_values=''
# XGB模型文件：train_data.csv
# Line模型文件：test_model.csv
file_path = Path(__file__).parent / "train_data.csv"

# 加载模型
# 加载LinearRegression模型
#model = LinearRegression()
#loaded_model = joblib.load(Path(__file__).parent / 'linear_regression_model_313.pkl')

# model = xgb.XGBRegressor()
model = joblib.load(Path(__file__).parent / 'jm_XGBClassifier_model_python38.pkl')
#model=[]


#result_str=''
# 定义服务器逻辑
def server(input: Inputs, output: Outputs, session: Session):
    @render.text
    def result_y_red_text():
        return result_y_pred_str.get()

    @render.text
    def result_text():
        return result_str.get()

    # 输出一个占位符文本（可选，根据实际需求替换）
    #df = reactive.value(1)#定义变量
    df_style=reactive.value(1)#1：默认是取测试集文件第1行记录 2：测试值文件所有记录 3：上传文件
    result_str = reactive.value('')
    result_y_pred_str = reactive.value('')

    @reactive.Effect
    @reactive.event(input.generate)#监控按钮动作作图
    def generate():
        print('单击generate按钮')
        global df
        df = read_excel()
        df_style.set(1)
        load_model()
        pass
    @reactive.Effect
    @reactive.event(input.generate_mul)  # 监控按钮动作作图
    def generate_mul():
        print('单击generate_mul按钮')
        global df
        df = read_excel()
        #df = read_excel()
        df_style.set(2)
        load_model()
        pass

    @reactive.Effect
    @reactive.event(input.generate_upload)  # 监控按钮动作作图
    def generate_upload():
        print('单击generate_upload按钮')
        global df
        df = parsed_file()
        if df.empty:
            ui.notification_show("请上传文件 ", duration=2,type='error')
            print('请上传文件')
            return
        df_style.set(3)
        load_model()
        pass



    @reactive.calc
    def parsed_file( i=0):#文件上传
        file: list[FileInfo] | None = input.upload_file()
        if file is None:

            # if(i!=1):
            #     ui.notification_show("请上传文件 ", duration=2,type='error')
            #     print('请上传文件')
            #print(pd.DataFrame())
            return pd.DataFrame()
            #exit()
            #assert "error"
            #return pd.DataFrame()
        return pd.read_csv(file[0]["datapath"])
        #return pd.read_excel(file[0]["datapath"], index_col=None)#, engine='openpyxl'
        #return pd.read_csv(file[0]["datapath"])

    @render.table
    def summary():#文件上传后展现表格
        global df
        #print('展现表格')
        df = parsed_file()
        #print('df的值')
        if df.empty:
            #print('初始化时空表格')
            return pd.DataFrame()
        #df.set(df_file)
        #df_style.set(3)
        #print('更新df'.df.get().count())

        #scatter_plot2(2)



        # Get the row count, column count, and column names of the DataFrame
        row_count = df.shape[0]
        column_count = df.shape[1]
        names = df.columns.tolist()
        column_names = ", ".join(str(name) for name in names)

        # Create a new DataFrame to display the information
        info_df = pd.DataFrame(
            {
                "Row Count": [row_count],
                "Column Count": [column_count],
                "Column Names": [column_names],
            }
        )

        # input.stats() is a list of strings; subset the columns based on the selected
        # checkboxes

        return info_df.loc[:, input.stats()]


    #@output
    @render.text
    def dynamic_inputs_placeholder():#初始化时，获取测试集文件栏目，自动生成文本框，并填充第一行记录
        print('初始化文本框')
        global df

        df = read_excel()

        if df is not None:
            # 创建一个包含所有文本输入框的行，每个输入框在一个列中
            columns = []
            global NUM_FEATURES
            NUM_FEATURES= len(df.columns)-1#一共有多少列特征


            print('实际目标值=',df[df.columns[-1]].iloc[0])
            ui.update_numeric("act_y",value=float(df[df.columns[-1]].iloc[0]),min=1, max=100)
            #print('这里更新了t')
            #for col in df.columns:
            #row_indices = range(len(df))  # 示例行索引
            #for i, col in zip(row_indices, df.columns):  # 同时迭代行索引和列名
            for i,col in enumerate(df.columns[:-1]):
                #print(f"Row index: {i}, Column: {col}")  # 调试输出

                # 动态分配列宽
                col_width =2

                column_ui= ui.row(
                    #col_width,
                    ui.input_numeric(
                        id=f"feature_{i}",
                        #id=input_id,
                        label=col,
                        value=float(df[col].iloc[0]),
                         min=1,
                        max=100
                    )# if not pd.isnull(df[col].iloc[0]) else "50"

                )
                columns.append(column_ui)
            inputs_row = ui.row(columns)  # 使用flex布局确保输入框可以换行 style={"display": "flex", "flex-wrap": "wrap"}

            return inputs_row
        else:
            return ui.p("No data to display.")


        # return "动态输入区域（此处为占位符）"


    # 创建一个反应式值来存储生成的数据

    #@reactive.event(input.generate)  # 当按钮被点击时触发
    @reactive.effect
    def _():
        load_model()
        print('全局初始化')


    @reactive.Calc
    def read_excel():
        # 读取Excel文件        # 读取“测试学生原始成绩”工作表
        #file_path = Path(__file__).parent / 'test_model.xlsx'
        #df_scores = pd.read_excel(file_path, sheet_name='测试学生原始成绩test', nrows=5, index_col=None)  # ,nrows=96


        df_scores = pd.read_csv(file_path)  # ,nrows=96
        return df_scores

    def get_data(i):
        #global NUM_FEATURES
        #global df
        global X
        global y
        print('特征值数量：', NUM_FEATURES)

        if i==1:#预测方式1:默认取出xls文件第一个测试值
            print('预测方式1')
            features = []
            y = np.array([float(input.act_y())])  # 实际目标值
            #print('act_y=', y)
            for i in range(0, NUM_FEATURES):
                feature_value = float(input[f'feature_{i}']())  # 动态获取输入框的值
                features.append(feature_value)
            #X = np.array([features]).reshape(1, -1)# 预测方式一（（默认只取一个测试值）

            #print('act_y=',input.act_y())
            #print('@@@@@',df.columns[-1])
            #exit()
            #y = pd.DataFrame([float(input.act_y())], columns=df.columns[-1])
            X = pd.DataFrame([features], columns=df.columns[:-1])

            #feature_names = X.columns
            #X = pd.DataFrame({features})
            #y = pd.DataFrame({float(input.act_y())})

        if i == 2:  # 预测方式2：取出默认xls文件所有数据集
            print('预测方式2',type(df))
            X = df.iloc[:, :-1]  # 取10个指标
            y = df.iloc[:, -1]  # 总分是最后一列

        if i == 3: # 预测方式3：取出上传xls文件所有数据集
            print('预测方式3')
            #print(df.count())
            X = df.iloc[:, :-1]  # 取10个指标
            y = df.iloc[:, -1]  # 总分是最后一列


        return X,y


    def load_model():#加载模型
        print('load_model===============,df_style=',df_style.get())

        # 加载xgboost模型
        # loaded_model = xgb.XGBRegressor()
        # # loaded_model = xgb.XGBRegressor(booster='gbtree', n_estimators=100, learning_rate=0.1, max_depth=10, subsample=1.0, colsample_bytree=1.0, min_child_weight=1.0, gamma=0.0,reg_lambda=1.0,alpha=0.0 )#lambda=1.0
        # loaded_model.load_model(Path(__file__).parent /'xgboost_model.model')
        #


        #print('开始做图：df_style=', df_style.get())
        global X,y,y_pred
        global shape,model
        global shap_values
        X, y = get_data(df_style.get())  # 根据df_style类型取出数据

        #print('测试值X=', X, '\n 实际值y=', y, '\n')

        y_pred = model.predict(X)  # //.flatten() 预测值

        # 计算SHAP值
        shape = shap.TreeExplainer(model)
        shap_values = shape(X)
        #shap_values = shape.shap_values(X)
        #shap_values = shape.explainer(X, fixed_context=1)
        # 计算 baseline（预测值的均值）
        baseline = np.mean(y_pred)  # 假设 pfun 是模型预测函数

        #print('指标X=',X,'/预测值y_pred=', y_pred)
        # ui.update_text("pre_y", value=str(y_pred))
        # 计算详细的评价指标
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mse)
        # r2 = r2_score(y, y_pred)

        y_pred_str = ''
        # ui.update_text('',value=y_pred)
        if int(df_style.get()) == 1:
            result_y_pred_str.set(y_pred)
            y_pred_str = f'Y_pred: {y_pred}\n'
        # global result_str
        result_str.set(
            f'{y_pred_str}Mean Squared Error (MSE): {mse}\nMean Absolute Error (MAE): {mae}\nRoot Mean Squared Error (RMSE): {rmse}')
        # print(result_str.get())
        # ui.update_text('result_text', value=result_str)
        # 输出评价指标
        # print(f'Mean Squared Error (MSE): {mse}')
        # print(f'Mean Absolute Error (MAE): {mae}')
        # print(f'Root Mean Sq   uared Error (RMSE): {rmse}')

    def scatter_plot2():


        # 创建一个图形和两个子图
        fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(5, 1)# figsize=(10, 45), dpi=50

# sns绘制数据集直方图 ==================================
        #print('df的类型=',type(df),df)
        #print(df['考试总分'])#df.iloc[:, -1]

        sns.histplot(y, bins=10, kde=True,ax=ax1)  # kde=False 表示绘制核密度估计曲线
        # 添加标题和标签
        ax1.set_title('因变量直方图')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        # 显示图表
        #ax1.tight_layout()
        #ax1.show()

        # ax1.scatter(y, y_pred, alpha=0.5)
        # ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 对角线
        # ax1.set_xlabel('Actual')
        # ax1.set_ylabel('Predicted')
        # ax1.set_title('Actual vs Predicted 散点图')

        #ax1.legend()
# 使用sns创建热力图===================================
        scaler = MinMaxScaler()
        data_array_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        sns.heatmap(data_array_normalized.corr(), annot=True, cmap='YlGnBu', fmt=".2f",ax=ax2)  # fmt=".2f" 用于保留两位小数
        ax2.set_title('相关系数热力图')


# sns绘制预测值与真实值的对比图 散点图 ==================================

        data = pd.DataFrame({
            'True Values': y,
            'Predicted Values': y_pred
        })
        sns.scatterplot(x='True Values', y='Predicted Values', data=data, alpha=0.7,ax=ax3)
        # 添加对角线（理想情况下，预测值等于真实值）
        min_val, max_val = plt.xlim()
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        # 设置标题和标签
        ax3.set_title('预测值与真实值的对比（散点图）')
        ax3.set_xlabel('真实值')
        ax3.set_ylabel('预测值')

# 绘制预测值与真实值的对比图  线图=================================
        data = pd.DataFrame({
            'True Values': y,
            'Predicted Values': y_pred,
            'x': range(len(y))
        })
        # print(data)
        # exit()
        # data['x'] = range(len(y_test))
        # plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='x', y='True Values', label='真实值' ,ax=ax4)#color='blue'
        sns.lineplot(data=data, x='x', y='Predicted Values', label='预测值', linestyle='--',ax=ax4)
        # 添加图例和标题
        ax4.legend(title='图例')
        ax4.set_title('真实值与预测值对比图')
        # # 残差图
        # residuals = y - y_pred
        # #ax2.figure(figsize=(8, 6))
        # sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
        # ax2.axhline(y=0, color='r', linestyle='--')
        # ax2.set_xlabel('Predicted')
        # ax2.set_ylabel('Residuals')
        # ax2.set_title('Residuals Plot 残差图')
        # #ax2.legend()
        # #plt.show()
        #
        #
#  绘制预测值与真实值的残差图 =================================
        data = pd.DataFrame({
            'Predicted Values': y_pred,
            'residuals': float(y )- float(y_pred),
            'x': range(len(y))

        })
        # print(data)
        # exit()
        # data['x'] = range(len(y_test))

        sns.scatterplot(x='Predicted Values', y='residuals', data=data, alpha=0.7,ax=ax5)
        # 在 y=0 的位置绘制一条水平线
        ax5.axhline(y=0, linestyle='--')#color='r'
        # 添加对角线（理想情况下，预测值等于真实值）
        # min_val, max_val = plt.xlim()
        # plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        ax5.set_title('预测值与真实值的残差图')
        # plt.show()
        # # 分布图
        # #ax3.figure(figsize=(8, 6))
        # sns.histplot(y_pred, kde=True, bins=30, label='Predicted')
        # # sns.histplot(residuals, kde=True, bins=30, label='Residuals', color='orange')  # 如果也想看残差分布
        # ax4.set_xlabel('Value')
        # ax4.set_ylabel('Density')
        # ax4.set_title('Distribution of Predicted Values 分布图')
        #ax3.legend()
        #plt.show()

        # 调整布局
        #plt.tight_layout()
        # 调整子图之间的间距
        #fig.subplots_adjust(hspace=0.5)
        return fig

    # 渲染图表
    # @output
    @render.plot  # 确保使用 render.plot 来处理 "plot" 类型的输出
    def scatter_plot_shap1():#单个样本时 瀑布图
        df_style.get()
        #print('scatter_plot_shap1')
        #return
        if len(X)!=1:
            pass
        #fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        explanation = shap.Explanation(shap_values.values[0], shape.expected_value, X.iloc[0],feature_names=df.columns.tolist())
        shap.plots.waterfall(explanation, show=False)
        plt.title("Feature Importance (WaterFall Plot)")
        #return fig

    @render.plot  # 确保使用 render.plot 来处理 "plot" 类型的输出
    def scatter_plot_shap2():#蜂群图
        df_style.get()
        explanation = shap.Explanation(shap_values, shape.expected_value, X,X.columns.tolist())
        shap.plots.beeswarm(explanation, show=False, plot_size=None)  # ,ax=ax[0]
        plt.title("Feature Importance (Bar Plot)", fontsize=16)
        #return fig

    @render.ui
    def scatter_plot_shap_text():
        #
        # 修改了源代码：D:\Python\Python313\Lib\site-packages\shap\plots\_text.py
        # 第334行“for i, token in enumerate(tokens):”内加入了：token=str(token)
        # 第319行“encoded_tokens”，加入了str(t)

        #str=""
        if len(X)!=1:
            pass
        df_style.get()

        str = ui.markdown(shap.plots.text(shap_values[0:3], display=False))

        # str=ui.markdown("""
        #            fdasfs
        #            <p>fda</p>
        #            <a href='#'>sss</a>
        #             """)

        return str

    @render.plot  # 确保使用 render.plot 来处理 "plot" 类型的输出
    def scatter_plot_shap4():  #
        #df_style.get()
        #shap_e = shape(X)

        #print('type=',type(shap_values[0]))
        #shap.plots.text(shap_values)

        #shap.plots.text(shap_values[0])
        plt.title("Feature Importance (Bar Plot)")



    @render.plot  # 确保使用 render.plot 来处理 "plot" 类型的输出
    def scatter_plot_shap3():#force力图
        df_style.get()
        shap.plots.force(shape.expected_value,shap_values.values[0],X.iloc[0],X.columns.tolist(),show=False ,matplotlib=True)
        # shap.plots.force(explanation)#,matplotlib=True
        plt.title("Feature Importance (Force Plot)")
        plt.tight_layout()



        #print('scatter_plot_shap')
        #n=df_style.get()==1?1:
        #lamda:



        # if int(df_style.get()) == 0:
        #     print('0000')
        # #pass
        #nrows=(lambda : 2 if int(df_style.get()) ==1 else 1)
        # if int(df_style.get()) ==1:#单个
        #     nrows=2
        # else:
        #     nrows=1

        #fig, axes= plt.subplots(nrows, 1,figsize=(16, 8))  # figsize=(10, 45), dpi=50
        #print('shap_values type=',shape(shap_values),'/, shape  =',shape(X))
        #return
        #ax1, ax2 ,ax3= axes


        #shap.plots.beeswarm(explanation, ax=axes[1], show=False,plot_size=None)
        #axes[0] = shap.summary_plot(shap_values, X, feature_names=df.columns, plot_type="bar",show=False)
        #axes[1] = shap.summary_plot(shap_values, X, feature_names=df.columns, plot_type="bar",show=False)
 # plot_type="violin"
 #
 #
 #        if len(X)==1:#单个样本时 瀑布图
 #            fig, axes = plt.subplots(2, 1, figsize=(16, 8))
 #            explanation = shap.Explanation(shap_values[0], shape.expected_value, X.iloc[0],feature_names=df.columns.tolist())
 #            axes[1] = shap.plots.waterfall(explanation, show=False)
 #            axes[1].set_title("Feature Importance (WaterFall Plot)")
 #        else:
 #
 #            # 1绘制 bar 图
 #            explanation = shap.Explanation(values=shap_values, base_values=shape.expected_value, data=X,
 #                                           feature_names=X.columns.tolist())
 #            shap.plots.beeswarm(explanation, ax=axes, show=False, plot_size=None)  # ,ax=ax[0]
 #            axes.set_title("Feature Importance (Bar Plot)", fontsize=16)
 #





        # ax.set_ylabel('Feature')
        #plt.tight_layout()


        #axes[0]=shap.summary_plot(shap_values, X, feature_names=df.columns, plot_type="violin", show=False)  # plot_type="violin"
        #axes[0].set_title("Feature Importance (Bar Plot)", fontsize=16)


        #print('df.columns=',df.columns,'/shap_values[0]=',shap_values[0],'/X.iloc[0]',X.iloc[0])
        if len(X)>1:#必须多个样本
            pass
            # shap.summary_plot(shap_values, X, feature_names=df.columns, plot_type="violin",
            #                   show=False,ax=axes[1])  # plot_type="violin"
            # axes[1].set_title("Feature Importance (violin Plot)", fontsize=16)
        # else:#单个样本
        #     explanation = shap.Explanation(shap_values[0], shape.expected_value,X.iloc[0],
        #                                    feature_names=df.columns.tolist(),ax=axes[1])
        #     shap.plots.waterfall(explanation, show=False)
        #     axes[1].set_title("Feature Importance (WaterFall Plot)")
        #     # ax.set_ylabel('Feature')
        #     plt.tight_layout()


        #return fig

    # 渲染图表
    #@output
    @render.plot  # 确保使用 render.plot 来处理 "plot" 类型的输出
    def scatter_plot():

        #global df
        print('作图开始catter_plot')

        #print('y=',y)
        # 创建一个图形和两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1)  # figsize=(10, 45), dpi=50

        sns.histplot(df.iloc[:, -1] ,bins=10, kde=True, ax=ax1)  # kde=False 表示绘制核密度估计曲线
        # 添加标题和标签
        ax1.set_title('因变量直方图')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        # 显示图表
        # ax1.tight_layout()
        # ax1.show()

        # ax1.scatter(y, y_pred, alpha=0.5)
        # ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 对角线
        # ax1.set_xlabel('Actual')
        # ax1.set_ylabel('Predicted')
        # ax1.set_title('Actual vs Predicted 散点图')

        # ax1.legend()
        # 使用sns创建热力图===================================
        scaler = MinMaxScaler()
        data_array_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        sns.heatmap(data_array_normalized.corr(), annot=True, cmap='YlGnBu', fmt=".2f", ax=ax2)  # fmt=".2f" 用于保留两位小数
        ax2.set_title('相关系数热力图')
        #fig=scatter_plot2()
        return fig

    @render.plot  # 确保使用 render.plot 来处理 "plot" 类型的输出
    def scatter_plot_pre():

        #global X,y
        print('scatter_plot_pre')
        #abc=df_style.get()
        #print('abc='.abc)
        if int(df_style.get()) ==0:
            pass

        fig, (ax1, ax2,ax3) = plt.subplots(3, 1)  # figsize=(10, 45), dpi=50
        # sns绘制预测值与真实值的对比图 散点图 ==================================

        data = pd.DataFrame({
            'True Values': y,
            'Predicted Values': y_pred
        })
        sns.scatterplot(x='True Values', y='Predicted Values', data=data, alpha=0.7, ax=ax1)
        # 添加对角线（理想情况下，预测值等于真实值）
        min_val, max_val = plt.xlim()
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        # 设置标题和标签
        ax1.set_title('预测值与真实值的对比（散点图）')
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')

        # 绘制预测值与真实值的对比图  线图=================================
        data = pd.DataFrame({
            'True Values': y,
            'Predicted Values': y_pred,
            'x': range(len(y))
        })
        # print(data)
        # exit()
        # data['x'] = range(len(y_test))
        # plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='x', y='True Values', label='真实值', ax=ax2)  # color='blue'
        sns.lineplot(data=data, x='x', y='Predicted Values', label='预测值', linestyle='--', ax=ax2)
        # 添加图例和标题
        ax2.legend(title='图例')
        ax2.set_title('真实值与预测值对比图')
        # # 残差图
        # residuals = y - y_pred
        # #ax2.figure(figsize=(8, 6))
        # sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
        # ax2.axhline(y=0, color='r', linestyle='--')
        # ax2.set_xlabel('Predicted')
        # ax2.set_ylabel('Residuals')
        # ax2.set_title('Residuals Plot 残差图')
        # #ax2.legend()
        # #plt.show()
        #
        #
        #  绘制预测值与真实值的残差图 =================================
        data = pd.DataFrame({
            'Predicted Values': y_pred,
            'residuals': y - y_pred,
            'x': range(len(y))

        })
        # print(data)
        # exit()
        # data['x'] = range(len(y_test))

        sns.scatterplot(x='Predicted Values', y='residuals', data=data, alpha=0.7, ax=ax3)
        # 在 y=0 的位置绘制一条水平线
        ax3.axhline(y=0, linestyle='--')  # color='r'
        # 添加对角线（理想情况下，预测值等于真实值）
        # min_val, max_val = plt.xlim()
        # plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        ax3.set_title('预测值与真实值的残差图')
        #pass
        #global df
        #print('作图开始scatter_plot')
        #return
        #fig=scatter_plot2()
        return fig

# 创建并运行应用
app = App(app_ui, server)
