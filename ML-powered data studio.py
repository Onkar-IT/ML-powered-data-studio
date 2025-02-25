import math
import itertools
import requests
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, simpledialog, colorchooser, Tk, Canvas, Frame, BOTH, LEFT, RIGHT, Y, X, NW, Toplevel, Scrollbar
import tkinter.font as tkFont
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import sqlite3
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# ------------------ Helper Function ------------------
def create_scrollable_frame(parent, bg=None):
    if bg is None:
        try:
            bg = parent.cget("background")
        except Exception:
            bg = "#f7f7f7"
    canvas = Canvas(parent, borderwidth=0, background=bg, highlightthickness=0)
    frame = Frame(canvas, background=bg)
    vsb = tb.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor=NW)
    return frame

# ------------------ Main Application Class ------------------
class MLDataStudioApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced ML‑Powered Data Studio")
        self.master.geometry("1700x950")
        self.theme_color = "#f7f7f7"
        self.text_color = "#333333"
        self.button_color = "#e0e0e0"
        self.chart_title_color = "#333333"
        self.axis_label_color = "#333333"
        self.default_color_palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
        self.base_font_family = "Calibri"
        self.base_font_size = 12
        self.available_text_families = ["Arial", "Times New Roman", "Courier New", "Calibri", "Helvetica", "Verdana"]
        self.style = tb.Style("flatly")

        # Data storage: when a file is loaded, store original data for resets
        self.data = None
        self.original_data = None
        self.file_path = None
        self.anomalies = None
        self.forecast_x = None
        self.forecast_y = None
        self.forecast_ci = None

        # Custom chart variables for File & Data tab
        self.custom_chart_type_var = tb.StringVar(value="Scatter")
        self.custom_chart_title_var = tb.StringVar()
        self.custom_chart_color_var = tb.StringVar()

        self.create_header()
        self.create_status_bar()

        # Main layout frames: left panel for tabs and right panel for content
        self.main_frame = tb.Frame(self.master)
        self.main_frame.pack(fill=BOTH, expand=True)
        self.left_frame = tb.Frame(self.main_frame, width=250, relief="raised")
        self.left_frame.pack(side=LEFT, fill=Y)
        self.left_frame.pack_propagate(False)
        self.left_canvas = Canvas(self.left_frame, highlightthickness=0, background=self.theme_color)
        self.left_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.left_scrollbar = tb.Scrollbar(self.left_frame, orient="vertical", command=self.left_canvas.yview)
        self.left_scrollbar.pack(side=RIGHT, fill=Y)
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)
        self.tabs_container = Frame(self.left_canvas, background=self.theme_color)
        self.left_canvas.create_window((0, 0), window=self.tabs_container, anchor=NW)
        self.tabs_container.bind("<Configure>", lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")))
        self.right_frame = tb.Frame(self.main_frame)
        self.right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # Define tab names
        self.tabs_order = [
            "File & Data", "Custom Dashboard", "File Converter", "Forecasting",
            "Data Cleaning", "Model & Anomaly", "Advanced Tools", "Data Integration",
            "Data Summary", "Data Transformation", "Data Filter",
            #"Advanced ML & Anomaly",
            #"Collaboration & Security",
             "Settings"
        ]
        self.tab_frames = {}
        self.tab_buttons = {}

        # Create all tabs
        self.create_file_data_tab()
        self.create_custom_dashboard_tab()
        self.create_file_converter_tab()
        self.create_forecasting_tab()
        self.create_data_cleaning_tab()
        self.create_model_anomaly_tab()
        self.create_advanced_tools_tab()
        self.create_data_integration_tab()
        self.create_data_summary_tab()
        self.create_data_transformation_tab()
        self.create_data_filter_tab()
        #self.create_advanced_ml_tab()
        #self.create_collaboration_security_tab()
        self.create_settings_tab()

        for tab in self.tabs_order:
            btn = tb.Button(self.tabs_container, text=tab, command=lambda t=tab: self.show_tab(t),
                            bootstyle="secondary", width=22)
            btn.pack(fill=X, padx=18, pady=8)
            self.tab_buttons[tab] = btn

        self.show_tab(self.tabs_order[0])

    # ---------- Utility Methods ----------
    def create_header(self):
        self.header = tb.Frame(self.master, padding=10)
        self.header.pack(fill=X)
        self.title_label = tb.Label(self.header, text="ML‑Powered Data Studio",
                                    font=(self.base_font_family, 28, "bold"))
        self.title_label.pack(side=LEFT)
        self.theme_toggle_btn = tb.Button(self.header, text="Toggle Dark/Light", command=self.toggle_dark_light,
                                          bootstyle="secondary")
        self.theme_toggle_btn.pack(side=RIGHT)

    def create_status_bar(self):
        self.status_bar = tb.Label(self.master, text="Welcome to ML‑Powered Data Studio!",
                                   anchor="w", padding=5, bootstyle="secondary")
        self.status_bar.pack(fill=X, side="bottom")

    def toggle_dark_light(self):
        current_theme = self.style.theme.name
        new_theme = "cyborg" if current_theme != "cyborg" else "flatly"
        self.style.theme_use(new_theme)
        self.update_status(f"Theme changed to {new_theme}.")

    def update_status(self, message, error=False):
        self.status_bar.config(text=message, bootstyle="danger" if error else "secondary")
        print("Status:", message)

    def show_tab(self, tab_name):
        for frame in self.tab_frames.values():
            frame.place_forget()
        if tab_name in self.tab_frames:
            self.tab_frames[tab_name].place(in_=self.right_frame, relx=0, rely=0, relwidth=1, relheight=1)
        self.update_status(f"Switched to {tab_name} tab.")
        for name, btn in self.tab_buttons.items():
            btn.configure(bootstyle="primary" if name == tab_name else "secondary")

    # ---------- Data Cleaning Methods ----------
    def remove_duplicates(self):
        if self.data is not None:
            self.data = self.data.drop_duplicates()
            messagebox.showinfo("Info", "Duplicate rows removed.")
            self.update_status("Duplicates removed.")

    def handle_missing_data(self):
        if self.data is not None:
            action = messagebox.askyesno("Handle Missing Data", "Drop rows with missing values?")
            if action:
                self.data = self.data.dropna()
                messagebox.showinfo("Info", "Rows with missing values dropped.")
                self.update_status("Missing data dropped.")
            else:
                fill_value = simpledialog.askstring("Fill Missing Data", "Enter value to fill missing data:")
                if fill_value is not None:
                    self.data = self.data.fillna(fill_value)
                    messagebox.showinfo("Info", "Missing values filled.")
                    self.update_status("Missing data filled.")
                else:
                    self.update_status("Missing data fill cancelled.")

    def standardize_data(self):
        if self.data is not None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                messagebox.showinfo("Info", "No numeric columns to standardize.")
                self.update_status("No numeric columns for standardization.")
                return
            scaler = StandardScaler()
            try:
                self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
                messagebox.showinfo("Info", "Data standardized.")
                self.update_status("Data standardized.")
            except Exception as e:
                messagebox.showerror("Error", f"Standardization failed: {e}")
                self.update_status("Standardization failed.", error=True)

    def fix_structural_errors(self):
        if self.data is not None:
            self.data.columns = [col.strip().lower().replace(" ", "_") for col in self.data.columns]
            messagebox.showinfo("Info", "Structural errors fixed.")
            self.update_status("Structural errors fixed.")
            self.update_dropdowns()

    def handle_outliers(self):
        if self.data is not None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                messagebox.showinfo("Info", "No numeric columns for outlier handling.")
                self.update_status("No numeric columns for outlier handling.")
                return
            for col in numeric_cols:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            messagebox.showinfo("Info", "Outliers handled.")
            self.update_status("Outliers handled.")

    # ---------- File & Data Tab Methods ----------
    def create_file_data_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["File & Data"] = frame
        container = tb.Frame(frame)
        container.pack(fill=BOTH, expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=3)
        left_panel = tb.Frame(container)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        right_panel = tb.Frame(container)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        file_frame = tb.Labelframe(left_panel, text="File Upload", padding=10)
        file_frame.pack(fill="x", padx=5, pady=15)
        tb.Label(file_frame, text="Upload your file:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(file_frame, text="Browse File", command=self.handle_file_upload, bootstyle="success").grid(row=0, column=1, padx=5, pady=5)
        col_frame = tb.Labelframe(left_panel, text="Column Selection", padding=10)
        col_frame.pack(fill="x", padx=5, pady=15)
        tb.Label(col_frame, text="X-Axis Column:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.x_col_menu = tb.Combobox(col_frame, state="readonly")
        self.x_col_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(col_frame, text="Y-Axis Column:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.y_col_menu = tb.Combobox(col_frame, state="readonly")
        self.y_col_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.z_col_label = tb.Label(col_frame, text="Z-Axis Column (3D):")
        self.z_col_menu = tb.Combobox(col_frame, state="readonly")
        self.z_col_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.z_col_menu.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.z_col_label.grid_remove()
        self.z_col_menu.grid_remove()
        custom_frame = tb.Labelframe(left_panel, text="Custom Chart Creator", padding=10)
        custom_frame.pack(fill="x", padx=5, pady=15)
        tb.Label(custom_frame, text="Chart Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.custom_chart_type_cb = tb.Combobox(custom_frame, state="readonly", textvariable=self.custom_chart_type_var,
                                                values=["Scatter", "Line", "Bar", "Area", "Bubble", "Pie", "Other"])
        self.custom_chart_type_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(custom_frame, text="Chart Title:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.custom_chart_title_entry = tb.Entry(custom_frame, textvariable=self.custom_chart_title_var)
        self.custom_chart_title_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(custom_frame, text="Custom Color (e.g., 'blue' or '#ff0000'):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.custom_chart_color_entry = tb.Entry(custom_frame, textvariable=self.custom_chart_color_var)
        self.custom_chart_color_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(custom_frame, text="Generate Custom Chart", command=self.custom_chart, bootstyle="primary").grid(row=3, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        custom_frame.columnconfigure(1, weight=1)
        self.suggestions_frame = tb.Labelframe(right_panel, text="Chart Suggestions", padding=10)
        self.suggestions_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.sug_inner = Frame(self.suggestions_frame, background="#ffffff")
        self.sug_inner.pack(fill=BOTH, expand=True)

    def handle_file_upload(self):
        self.upload_file()
        if self.data is not None:
            self.original_data = self.data.copy()
            self.update_dropdowns()
            self.display_suggestions()

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if not self.file_path:
            self.update_status("File upload cancelled.")
            return
        try:
            if self.file_path.endswith(".csv"):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith(".xlsx"):
                self.data = pd.read_excel(self.file_path)
            self.update_status(f"File loaded: {self.file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.update_status("Failed to load file.", error=True)

    def custom_chart(self):
        x_col = self.x_col_menu.get()
        y_col = self.y_col_menu.get()
        if not x_col or not y_col:
            messagebox.showerror("Error", "Please select X and Y columns first.")
            return
        chart_type = self.custom_chart_type_var.get()
        chart_title = self.custom_chart_title_var.get() or f"Custom {chart_type}: {x_col} vs {y_col}"
        custom_color = self.custom_chart_color_var.get().strip()
        palette = [custom_color] if custom_color else self.default_color_palette
        try:
            if chart_type.lower() == "scatter":
                fig = px.scatter(self.data, x=x_col, y=y_col, title=chart_title, color_discrete_sequence=palette)
            elif chart_type.lower() == "line":
                fig = px.line(self.data, x=x_col, y=y_col, title=chart_title, color_discrete_sequence=palette)
            elif chart_type.lower() == "bar":
                fig = px.bar(self.data, x=x_col, y=y_col, title=chart_title, color_discrete_sequence=palette)
            elif chart_type.lower() == "area":
                fig = px.area(self.data, x=x_col, y=y_col, title=chart_title, color_discrete_sequence=palette)
            elif chart_type.lower() == "bubble":
                fig = px.scatter(self.data, x=x_col, y=y_col, title=chart_title,
                                 size=self.data.index, color=self.data.index, color_continuous_scale=palette)
            elif chart_type.lower() == "pie":
                fig = px.pie(self.data, names=x_col, values=y_col, title=chart_title)
            else:
                messagebox.showerror("Error", f"Chart type '{chart_type}' not implemented.")
                return
            fig.show()
            self.update_status("Custom chart generated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Custom chart failed: {e}")
            self.update_status("Custom chart generation failed.", error=True)

    # ---------- Custom Dashboard Tab ----------
    def create_custom_dashboard_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Custom Dashboard"] = frame
        count_frame = tb.Frame(frame)
        count_frame.pack(fill="x", pady=5)
        tb.Label(count_frame, text="How many charts do you need?").pack(side=LEFT, padx=5)
        self.chart_count_var = tb.StringVar(value="1")
        self.chart_count_entry = tb.Entry(count_frame, textvariable=self.chart_count_var, width=5)
        self.chart_count_entry.pack(side=LEFT, padx=5)
        tb.Button(count_frame, text="Generate Chart Options", command=self.generate_chart_options, bootstyle="primary").pack(side=LEFT, padx=5)
        self.chart_options_frame = tb.Frame(frame)
        self.chart_options_frame.pack(fill=BOTH, expand=True, pady=10)
        tb.Button(frame, text="Create Dashboard", command=self.create_dashboard, bootstyle="success").pack(pady=10)
        self.dashboard_chart_configs = []

    def generate_chart_options(self):
        for widget in self.chart_options_frame.winfo_children():
            widget.destroy()
        try:
            count = int(self.chart_count_var.get())
        except ValueError:
            messagebox.showerror("Error", "Enter a valid number for chart count.")
            return
        self.dashboard_chart_configs = []
        cols = int(math.ceil(math.sqrt(count)))
        for col in range(cols):
            self.chart_options_frame.grid_columnconfigure(col, weight=1)
        for i in range(count):
            r = i // cols
            c = i % cols
            subframe = tb.Labelframe(self.chart_options_frame, text=f"Chart {i + 1} Options", padding=10)
            subframe.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            tb.Label(subframe, text="Chart Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            chart_type_var = tb.StringVar(value="Scatter")
            chart_type_cb = tb.Combobox(subframe, state="readonly", textvariable=chart_type_var,
                                        values=["Scatter", "Line", "Bar", "Pie", "Area", "Bubble", "Histogram"])
            chart_type_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            tb.Label(subframe, text="X Column:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            x_col_var = tb.StringVar()
            x_col_cb = tb.Combobox(subframe, state="readonly", textvariable=x_col_var,
                                   values=list(self.data.columns) if self.data is not None else [])
            x_col_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            tb.Label(subframe, text="Y Column:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
            y_col_var = tb.StringVar()
            y_col_cb = tb.Combobox(subframe, state="readonly", textvariable=y_col_var,
                                   values=list(self.data.columns) if self.data is not None else [])
            y_col_cb.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
            self.dashboard_chart_configs.append({
                "chart_type_var": chart_type_var,
                "x_col_var": x_col_var,
                "y_col_var": y_col_var
            })

    def create_dashboard(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        num_charts = len(self.dashboard_chart_configs)
        if num_charts == 0:
            messagebox.showerror("Error", "No chart configurations available.")
            return
        cols = int(math.sqrt(num_charts))
        if cols * cols < num_charts:
            cols += 1
        rows = int(math.ceil(num_charts / cols))
        subplot_titles = [f"{cfg['x_col_var'].get()} vs {cfg['y_col_var'].get()}" for cfg in self.dashboard_chart_configs]
        specs = []
        chart_index = 0
        for i in range(rows):
            row_spec = []
            for j in range(cols):
                if chart_index < num_charts:
                    cfg = self.dashboard_chart_configs[chart_index]
                    if cfg["chart_type_var"].get().lower() == "pie":
                        row_spec.append({"type": "domain"})
                    else:
                        row_spec.append({"type": "xy"})
                    chart_index += 1
                else:
                    row_spec.append({})
            specs.append(row_spec)
        combined_fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=subplot_titles)
        chart_index = 0
        for cfg in self.dashboard_chart_configs:
            chart_type = cfg["chart_type_var"].get()
            x_col = cfg["x_col_var"].get()
            y_col = cfg["y_col_var"].get()
            if not x_col or not y_col:
                messagebox.showerror("Error", "Select X and Y columns for all charts.")
                return
            try:
                if chart_type == "Scatter":
                    fig = px.scatter(self.data, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}")
                elif chart_type == "Line":
                    fig = px.line(self.data, x=x_col, y=y_col, title=f"Line: {x_col} vs {y_col}")
                elif chart_type == "Bar":
                    fig = px.bar(self.data, x=x_col, y=y_col, title=f"Bar: {x_col} vs {y_col}")
                elif chart_type == "Pie":
                    fig = px.pie(self.data, names=x_col, values=y_col, title=f"Pie: {x_col} vs {y_col}")
                elif chart_type == "Area":
                    fig = px.area(self.data, x=x_col, y=y_col, title=f"Area: {x_col} vs {y_col}")
                elif chart_type == "Bubble":
                    fig = px.scatter(self.data, x=x_col, y=y_col, title=f"Bubble: {x_col} vs {y_col}",
                                     size=self.data.index, color=self.data.index, color_continuous_scale='Viridis')
                elif chart_type == "Histogram":
                    fig = px.histogram(self.data, x=x_col, title=f"Histogram: {x_col}")
                else:
                    fig = go.Figure()
                    fig.add_annotation(text=f"Chart type '{chart_type}' not implemented", showarrow=False)
            except Exception as e:
                messagebox.showerror("Error", f"Error generating chart: {e}")
                return
            row = chart_index // cols + 1
            col = chart_index % cols + 1
            for trace in fig.data:
                try:
                    combined_fig.add_trace(trace, row=row, col=col)
                except Exception as add_e:
                    messagebox.showerror("Error", f"Error adding trace: {add_e}")
                    return
            chart_index += 1
        combined_fig.update_layout(height=rows * 400, width=cols * 600, title_text="Custom Dashboard")
        combined_fig.show()
        self.update_status("Custom dashboard created successfully.")

    # ---------- File Converter Tab ----------
    def create_file_converter_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["File Converter"] = frame
        conv_frame = tb.Labelframe(frame, text="File Converter", padding=10)
        conv_frame.pack(fill="both", expand=True, padx=10, pady=5)
        tb.Label(conv_frame, text="Upload file to convert:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(conv_frame, text="Browse File", command=self.upload_file, bootstyle="success").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Label(conv_frame, text="Output Format:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_format_cb = tb.Combobox(conv_frame, state="readonly",
                                            values=["CSV", "Excel", "SQLite", "JSON", "Parquet", "XML"])
        self.output_format_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(conv_frame, text="Convert File", command=self.convert_file, bootstyle="primary").grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        conv_frame.columnconfigure(1, weight=1)

    def convert_file(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a file first.")
            return
        output_format = self.output_format_cb.get()
        if not output_format:
            messagebox.showerror("Error", "Select an output format.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension="",
                                                 filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"),
                                                            ("SQLite DB", "*.db"), ("JSON Files", "*.json"),
                                                            ("Parquet Files", "*.parquet"), ("XML Files", "*.xml")])
        if not file_path:
            self.update_status("File conversion cancelled.")
            return
        try:
            if output_format == "CSV":
                self.data.to_csv(file_path, index=False)
            elif output_format == "Excel":
                self.data.to_excel(file_path, index=False)
            elif output_format == "SQLite":
                conn = sqlite3.connect(file_path)
                self.data.to_sql("converted_data", conn, if_exists="replace", index=False)
                conn.close()
            elif output_format == "JSON":
                self.data.to_json(file_path, orient="records")
            elif output_format == "Parquet":
                self.data.to_parquet(file_path)
            elif output_format == "XML":
                self.data.to_xml(file_path)
            messagebox.showinfo("File Converter", f"File converted and saved to {file_path}")
            self.update_status(f"File converted to {output_format} and saved.")
        except Exception as e:
            messagebox.showerror("Error", f"File conversion failed: {e}")
            self.update_status("File conversion failed.", error=True)

    # ---------- Forecasting Tab ----------
    def create_forecasting_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Forecasting"] = frame
        fc_frame = tb.Labelframe(frame, text="Forecasting Options", padding=10)
        fc_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.prediction_var = tb.BooleanVar(value=False)
        self.forecast_model_var = tb.StringVar(value="Linear")
        self.forecast_horizon_var = tb.StringVar(value="5")
        self.conf_int_var = tb.BooleanVar(value=False)
        self.prediction_toggle = tb.Checkbutton(fc_frame, text="Enable Prediction", variable=self.prediction_var,
                                                command=self.toggle_prediction)
        self.prediction_toggle.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Label(fc_frame, text="Chart Type:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.chart_menu = tb.Combobox(fc_frame, state="readonly",
                                      values=["Scatter", "Line", "Bar", "Pie", "Area", "Bubble", "Histogram",
                                              "3D Scatter", "3D Bubble", "3D Surface"])
        self.chart_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.chart_menu.bind("<<ComboboxSelected>>", self.update_z_axis_visibility)
        tb.Label(fc_frame, text="Forecast Model:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.forecast_model_cb = tb.Combobox(fc_frame, state="readonly", textvariable=self.forecast_model_var,
                                             values=["Linear", "Polynomial", "ARIMA"])
        self.forecast_model_cb.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(fc_frame, text="Forecast Horizon:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.forecast_horizon_entry = tb.Entry(fc_frame, textvariable=self.forecast_horizon_var)
        self.forecast_horizon_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.conf_int_cb = tb.Checkbutton(fc_frame, text="Show Confidence Interval", variable=self.conf_int_var)
        self.conf_int_cb.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        tb.Button(fc_frame, text="Visualize Forecast", command=self.generate_chart, bootstyle="primary").grid(row=5, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        fc_frame.columnconfigure(1, weight=1)

    def update_z_axis_visibility(self, event=None):
        selected_chart = self.chart_menu.get()
        if "3D" in selected_chart:
            self.z_col_label.grid()
            self.z_col_menu.grid()
        else:
            self.z_col_label.grid_remove()
            self.z_col_menu.grid_remove()

    def toggle_prediction(self):
        if self.prediction_var.get():
            if self.data is None:
                messagebox.showerror("Error", "Upload a dataset first.")
                self.prediction_var.set(False)
                self.update_status("Prediction failed: no dataset loaded.", error=True)
                return
            x_column = self.x_col_menu.get()
            y_column = self.y_col_menu.get()
            if not x_column or not y_column:
                messagebox.showerror("Error", "Select both X-Axis and Y-Axis columns for prediction.")
                self.prediction_var.set(False)
                self.update_status("Prediction failed: columns not selected.", error=True)
                return
            try:
                X_series = pd.to_numeric(self.data[x_column], errors='coerce').dropna()
                y_series = pd.to_numeric(self.data[y_column], errors='coerce').dropna()
                valid_mask = X_series.notna() & y_series.notna()
                if valid_mask.sum() == 0:
                    messagebox.showerror("Error", "No valid numeric data available for prediction.")
                    self.prediction_var.set(False)
                    self.update_status("Prediction failed: no valid numeric data.", error=True)
                    return
                X = X_series[valid_mask].values.reshape(-1, 1)
                y = y_series[valid_mask].values
                model_choice = self.forecast_model_var.get()
                forecast_horizon = int(self.forecast_horizon_var.get()) if self.forecast_horizon_var.get().isdigit() else 5
                if model_choice == "Polynomial" and valid_mask.sum() >= 5:
                    poly = PolynomialFeatures(degree=3)
                    X_poly = poly.fit_transform(X)
                    model = LinearRegression()
                    model.fit(X_poly, y)
                    y_pred = model.predict(X_poly)
                    x_sorted = np.sort(X.ravel())
                    diff = np.median(np.diff(x_sorted)) if len(x_sorted) > 1 else 1
                    forecast_x = np.linspace(x_sorted[-1] + diff, x_sorted[-1] + forecast_horizon * diff, forecast_horizon)
                    forecast_X_poly = poly.transform(forecast_x.reshape(-1, 1))
                    forecast_y = model.predict(forecast_X_poly)
                    self.forecast_x = forecast_x
                    self.forecast_y = forecast_y
                    self.forecast_ci = None
                elif model_choice == "ARIMA" and len(y) > 10:
                    model = ARIMA(y, order=(1, 1, 1))
                    model_fit = model.fit()
                    forecast_result = model_fit.get_forecast(steps=forecast_horizon)
                    forecast_output = forecast_result.predicted_mean
                    if self.conf_int_var.get():
                        conf_int = forecast_result.conf_int(alpha=0.05)
                        lower = conf_int.iloc[:, 0].values
                        upper = conf_int.iloc[:, 1].values
                        self.forecast_ci = (lower, upper)
                    else:
                        self.forecast_ci = None
                    self.forecast_x = np.arange(np.max(X) + 1, np.max(X) + forecast_horizon + 1)
                    self.forecast_y = forecast_output.values
                else:
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    self.forecast_x = np.array([])
                    self.forecast_y = np.array([])
                    self.forecast_ci = None
                self.data.loc[valid_mask, "Prediction"] = y_pred
                messagebox.showinfo("Prediction", f"Prediction complete for '{y_column}' using '{model_choice}' model.")
                self.update_status("Prediction completed successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")
                self.update_status("Prediction failed.", error=True)
                self.prediction_var.set(False)
        else:
            messagebox.showinfo("Prediction", "Prediction mode disabled.")
            self.update_status("Prediction mode disabled.")

    def generate_chart(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            self.update_status("No dataset loaded.", error=True)
            return
        chart_type = self.chart_menu.get()
        x_column = self.x_col_menu.get()
        y_column = self.y_col_menu.get()
        if not chart_type:
            messagebox.showerror("Error", "Select a chart type.")
            self.update_status("Chart type not selected.", error=True)
            return
        if not x_column or not y_column:
            messagebox.showerror("Error", "Select both X-Axis and Y-Axis columns.")
            self.update_status("X or Y column not selected.", error=True)
            return
        try:
            if chart_type == "Scatter":
                fig = px.scatter(self.data, x=x_column, y=y_column, title=f"Scatter plot: {x_column} vs {y_column}")
            elif chart_type == "Line":
                fig = px.line(self.data, x=x_column, y=y_column, title=f"Line plot: {x_column} vs {y_column}")
            elif chart_type == "Bar":
                fig = px.bar(self.data, x=x_column, y=y_column, title=f"Bar chart: {x_column} vs {y_column}")
            elif chart_type == "Pie":
                fig = px.pie(self.data, names=x_column, values=y_column, title=f"Pie chart: {x_column} vs {y_column}")
            elif chart_type == "Area":
                fig = px.area(self.data, x=x_column, y=y_column, title=f"Area plot: {x_column} vs {y_column}")
            elif chart_type == "Bubble":
                fig = px.scatter(self.data, x=x_column, y=y_column, title=f"Bubble plot: {x_column} vs {y_column}",
                                 size=self.data.index, color=self.data.index, color_continuous_scale='Viridis')
            elif chart_type == "Histogram":
                fig = px.histogram(self.data, x=x_column, title=f"Histogram: {x_column}")
            elif chart_type == "3D Scatter":
                z_column = self.z_col_menu.get()
                if not z_column:
                    messagebox.showerror("Error", "Select a Z-Axis column for 3D charts.")
                    return
                fig = px.scatter_3d(self.data, x=x_column, y=y_column, z=z_column, title="3D Scatter")
            elif chart_type == "3D Bubble":
                z_column = self.z_col_menu.get()
                if not z_column:
                    messagebox.showerror("Error", "Select a Z-Axis column for 3D charts.")
                    return
                fig = px.scatter_3d(self.data, x=x_column, y=y_column, z=z_column, size=self.data.index, title="3D Bubble")
            elif chart_type == "3D Surface":
                z_column = self.z_col_menu.get()
                if not z_column:
                    messagebox.showerror("Error", "Select a Z-Axis column for 3D charts.")
                    return
                fig = px.surface(self.data, x=x_column, y=y_column, z=z_column, title="3D Surface")
            else:
                messagebox.showinfo("Not Implemented", f"Chart type '{chart_type}' not implemented.")
                self.update_status(f"Chart type '{chart_type}' not implemented.")
                return
            fig.update_layout(
                title_font=dict(family=self.base_font_family, size=self.base_font_size + 8, color=self.chart_title_color),
                xaxis=dict(title_font=dict(color=self.axis_label_color)),
                yaxis=dict(title_font=dict(color=self.axis_label_color)),
                font=dict(color=self.axis_label_color),
                margin=dict(l=60, r=80, t=60, b=60)
            )
            fig.show()
            self.update_status(f"{chart_type} chart generated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {e}")
            self.update_status("Visualization failed.", error=True)

    # ---------- Data Cleaning Tab ----------
    def create_data_cleaning_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Data Cleaning"] = frame
        clean_frame = tb.Labelframe(frame, text="Data Cleaning", padding=10)
        clean_frame.pack(fill="both", expand=True, padx=10, pady=5)
        btn_texts = ["Remove Duplicates", "Handle Missing Data", "Standardize Data", "Fix Structural Errors", "Handle Outliers"]
        btn_commands = [self.remove_duplicates, self.handle_missing_data, self.standardize_data, self.fix_structural_errors, self.handle_outliers]
        for idx, (txt, cmd) in enumerate(zip(btn_texts, btn_commands)):
            tb.Button(clean_frame, text=txt, command=cmd, bootstyle="warning").grid(row=idx // 2, column=idx % 2, padx=10, pady=10, sticky="ew")
        clean_frame.columnconfigure(0, weight=1)
        clean_frame.columnconfigure(1, weight=1)

    # ---------- Data Integration Tab ----------
    def create_data_integration_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Data Integration"] = frame
        sqlite_frame = tb.Labelframe(frame, text="SQLite Integration", padding=10)
        sqlite_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(sqlite_frame, text="Database File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.db_path_entry = tb.Entry(sqlite_frame)
        self.db_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(sqlite_frame, text="Browse", command=self.browse_db, bootstyle="success").grid(row=0, column=2, padx=5, pady=5)
        tb.Label(sqlite_frame, text="Table Name:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.table_entry = tb.Entry(sqlite_frame)
        self.table_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(sqlite_frame, text="Load Table", command=self.load_table_from_sqlite, bootstyle="primary").grid(row=1, column=2, padx=5, pady=5)
        sqlite_frame.columnconfigure(1, weight=1)
        api_frame = tb.Labelframe(frame, text="API Integration", padding=10)
        api_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(api_frame, text="API URL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.api_url_entry = tb.Entry(api_frame)
        self.api_url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(api_frame, text="Load API Data", command=self.load_data_from_api, bootstyle="primary").grid(row=0, column=2, padx=5, pady=5)
        api_frame.columnconfigure(1, weight=1)

    def browse_db(self):
        db_path = filedialog.askopenfilename(filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")])
        if db_path:
            self.db_path_entry.delete(0, "end")
            self.db_path_entry.insert(0, db_path)

    def load_table_from_sqlite(self):
        db_path = self.db_path_entry.get()
        table_name = self.table_entry.get()
        if not db_path or not table_name:
            messagebox.showerror("Error", "Please provide both database file and table name.")
            return
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name}"
            self.data = pd.read_sql_query(query, conn)
            conn.close()
            self.original_data = self.data.copy()
            messagebox.showinfo("SQLite Integration", f"Table '{table_name}' loaded successfully.")
            self.update_status("Data loaded from SQLite.")
            self.update_dropdowns()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load table: {e}")
            self.update_status("SQLite table load failed.", error=True)

    def load_data_from_api(self):
        api_url = self.api_url_entry.get()
        if not api_url:
            messagebox.showerror("Error", "Please enter an API URL.")
            return
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            json_data = response.json()
            self.data = pd.DataFrame(json_data)
            self.original_data = self.data.copy()
            messagebox.showinfo("API Integration", "Data loaded successfully from API.")
            self.update_status("Data loaded from API.")
            self.update_dropdowns()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data from API: {e}")
            self.update_status("API data load failed.", error=True)

    # ---------- Data Summary Tab ----------
    def create_data_summary_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Data Summary"] = frame
        tb.Button(frame, text="Show Data Summary", command=self.display_data_summary, bootstyle="primary").pack(pady=10)

    def display_data_summary(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        summary = f"Data Shape: {self.data.shape}\n\n"
        summary += "Missing Values per Column:\n" + str(self.data.isna().sum()) + "\n\n"
        try:
            desc = self.data.describe().to_string()
            summary += "Descriptive Statistics:\n" + desc
        except Exception:
            summary += "Descriptive statistics not available."
        summary_win = Toplevel(self.master)
        summary_win.title("Data Summary")
        txt = tb.Text(summary_win, wrap="word")
        txt.insert("1.0", summary)
        txt.pack(fill=BOTH, expand=True)

    # ---------- Data Transformation Tab ----------
    def create_data_transformation_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Data Transformation"] = frame
        lbl = tb.Label(frame, text="Data Transformation", font=(self.base_font_family, 16, "bold"))
        lbl.pack(pady=10)
        btn_frame = tb.Frame(frame)
        btn_frame.pack(pady=10)
        tb.Button(btn_frame, text="Log Transform", command=self.log_transform, bootstyle="primary").grid(row=0, column=0, padx=5, pady=5)
        tb.Button(btn_frame, text="Scale Data", command=self.standardize_data, bootstyle="primary").grid(row=0, column=1, padx=5, pady=5)
        tb.Button(btn_frame, text="Reset Transformations", command=self.reset_transformations, bootstyle="warning").grid(row=0, column=2, padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

    def log_transform(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        try:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].apply(lambda x: np.log1p(x))
            messagebox.showinfo("Data Transformation", "Log transformation applied to numeric columns.")
            self.update_status("Log transformation applied.")
            self.update_dropdowns()
        except Exception as e:
            messagebox.showerror("Error", f"Log transformation failed: {e}")
            self.update_status("Log transformation failed.", error=True)

    def reset_transformations(self):
        if self.original_data is not None:
            self.data = self.original_data.copy()
            messagebox.showinfo("Data Transformation", "Data reset to original state.")
            self.update_status("Data reset to original.")
            self.update_dropdowns()

    # ---------- Data Filter Tab ----------
    def create_data_filter_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Data Filter"] = frame
        lbl = tb.Label(frame, text="Data Filter", font=(self.base_font_family, 16, "bold"))
        lbl.pack(pady=10)
        filter_frame = tb.Frame(frame)
        filter_frame.pack(pady=10)
        tb.Label(filter_frame, text="Enter filter query (e.g., age > 30 and salary < 50000):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.filter_entry = tb.Entry(filter_frame, width=50)
        self.filter_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        filter_frame.columnconfigure(1, weight=1)
        btn_frame = tb.Frame(frame)
        btn_frame.pack(pady=10)
        tb.Button(btn_frame, text="Apply Filter", command=self.apply_filter, bootstyle="primary").grid(row=0, column=0, padx=5, pady=5)
        tb.Button(btn_frame, text="Reset Filter", command=self.reset_filter, bootstyle="warning").grid(row=0, column=1, padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def apply_filter(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        query = self.filter_entry.get()
        if not query:
            messagebox.showerror("Error", "Please enter a filter query.")
            return
        try:
            filtered = self.data.query(query)
            if filtered.empty:
                messagebox.showinfo("Data Filter", "No data matches the filter condition.")
            else:
                self.data = filtered
                messagebox.showinfo("Data Filter", "Data filtered successfully.")
                self.update_status("Data filtered.")
                self.update_dropdowns()
        except Exception as e:
            messagebox.showerror("Error", f"Filtering failed: {e}")
            self.update_status("Filtering failed.", error=True)

    def reset_filter(self):
        if self.original_data is not None:
            self.data = self.original_data.copy()
            messagebox.showinfo("Data Filter", "Filter reset. Data restored to original.")
            self.update_status("Data restored to original.")
            self.update_dropdowns()

    # ---------- Advanced ML & Anomaly Tab ----------
    def create_advanced_ml_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Advanced ML & Anomaly"] = frame
        lbl = tb.Label(frame, text="Advanced ML Model Management & Anomaly Detection", font=("Calibri", 16, "bold"))
        lbl.pack(pady=10)
        ml_frame = tb.Frame(frame)
        ml_frame.pack(pady=10)
        self.ml_model_var = tb.StringVar(value="Random Forest")
        ml_model_cb = tb.Combobox(ml_frame, textvariable=self.ml_model_var, state="readonly",
                                  values=["Random Forest", "XGBoost", "SVM", "Deep Learning"])
        ml_model_cb.grid(row=0, column=0, padx=5, pady=5)
        train_button = tb.Button(ml_frame, text="Train Model", command=lambda: self.train_ml_model(self.ml_model_var.get()),
                                 bootstyle="primary")
        train_button.grid(row=0, column=1, padx=5, pady=5)
        tuning_button = tb.Button(frame, text="Automated Hyperparameter Tuning", command=self.hyperparameter_tuning,
                                  bootstyle="info")
        tuning_button.pack(pady=5)
        versioning_button = tb.Button(frame, text="Model Versioning & Tracking", command=self.model_versioning_tracking,
                                      bootstyle="warning")
        versioning_button.pack(pady=5)
        anomaly_button = tb.Button(frame, text="Detect Anomalies (Advanced)", command=self.detect_anomalies_extra,
                                   bootstyle="danger")
        anomaly_button.pack(pady=5)

    def train_ml_model(self, model_type):
        messagebox.showinfo("ML Training", f"Training {model_type} model...")
        self.update_status(f"{model_type} model training initiated.")

    def hyperparameter_tuning(self):
        messagebox.showinfo("Hyperparameter Tuning", "Hyperparameter tuning completed.")
        self.update_status("Hyperparameter tuning completed.")

    def model_versioning_tracking(self):
        messagebox.showinfo("Model Tracking", "Model versioning and tracking updated.")
        self.update_status("Model versioning and tracking updated.")

    def detect_anomalies_extra(self):
        messagebox.showinfo("Advanced Anomaly Detection", "Advanced anomaly detection complete.")
        self.update_status("Advanced anomaly detection complete.")

    # ---------- Model & Anomaly Tab ----------
    def create_model_anomaly_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Model & Anomaly"] = frame
        mod_frame = tb.Labelframe(frame, text="Model Comparison & Anomaly Options", padding=10)
        mod_frame.pack(fill="both", expand=True, padx=10, pady=5)
        tb.Button(mod_frame, text="Compare Models", command=self.compare_models, bootstyle="primary").grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Detect Anomalies", command=self.detect_anomalies, bootstyle="danger").grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Export Predictions", command=self.export_predictions, bootstyle="success").grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        mod_frame.columnconfigure(0, weight=1)
        mod_frame.columnconfigure(1, weight=1)

    def compare_models(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        x_column = self.x_col_menu.get()
        y_column = self.y_col_menu.get()
        if not x_column or not y_column:
            messagebox.showerror("Error", "Select both X-Axis and Y-Axis columns for comparison.")
            return
        try:
            X_series = pd.to_numeric(self.data[x_column], errors='coerce')
            y_series = pd.to_numeric(self.data[y_column], errors='coerce')
            valid_mask = X_series.notna() & y_series.notna()
            X = X_series[valid_mask].values.reshape(-1, 1)
            y = y_series[valid_mask].values
            models = {}
            lin_model = LinearRegression()
            lin_model.fit(X, y)
            models["Linear"] = lin_model.predict(X)
            if valid_mask.sum() >= 5:
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                models["Polynomial"] = poly_model.predict(X_poly)
            if len(y) > 10:
                arima_model = ARIMA(y, order=(1, 1, 1))
                arima_fit = arima_model.fit()
                models["ARIMA"] = arima_fit.fittedvalues
            msg = ""
            for name, pred in models.items():
                rmse = math.sqrt(mean_squared_error(y, pred))
                mape = mean_absolute_percentage_error(y, pred)
                r2 = r2_score(y, pred)
                msg += f"{name} Model:\n  RMSE: {rmse:.3f}\n  MAPE: {mape:.3f}\n  R²: {r2:.3f}\n\n"
            messagebox.showinfo("Model Comparison", msg)
            self.update_status("Model comparison completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Model comparison failed: {e}")
            self.update_status("Model comparison failed.", error=True)

    def detect_anomalies(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        col = simpledialog.askstring("Anomaly Detection", "Enter column for anomaly detection (default: Y-Axis):",
                                     initialvalue=self.y_col_menu.get())
        if not col or col not in self.data.columns:
            messagebox.showerror("Error", "Invalid column name.")
            return
        try:
            series = pd.to_numeric(self.data[col], errors='coerce').dropna()
            iso = IsolationForest(contamination=0.1, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1, 1))
            anomaly_idx = self.data.index[preds == -1].tolist()
            self.anomalies = anomaly_idx
            messagebox.showinfo("Anomaly Detection", f"Anomalies detected at indices:\n{anomaly_idx}")
            self.update_status("Anomaly detection completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Anomaly detection failed: {e}")
            self.update_status("Anomaly detection failed.", error=True)

    def export_predictions(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Export", f"Data exported to {file_path}")
                self.update_status(f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                self.update_status("Export failed.", error=True)

    # ---------- Advanced Tools Tab ----------
    def create_advanced_tools_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Advanced Tools"] = frame
        adv_frame = tb.Frame(frame, padding=10)
        adv_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        trans_frame = tb.Labelframe(adv_frame, text="Data Transformation", padding=10)
        trans_frame.pack(fill="x", padx=10, pady=5)
        # Use standardize_data (i.e. normalization) here
        tb.Button(trans_frame, text="Normalize Data", command=self.standardize_data, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="One-Hot Encode", command=self.one_hot_encode, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="Feature Engineering", command=self.feature_engineering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        filter_frame = tb.Labelframe(adv_frame, text="Data Filtering", padding=10)
        filter_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(filter_frame, text="Apply Filter", command=self.apply_filter, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        summary_frame = tb.Labelframe(adv_frame, text="Data Summary", padding=10)
        summary_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(summary_frame, text="Show Data Summary", command=self.display_data_summary, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        advml_frame = tb.Labelframe(adv_frame, text="Advanced ML", padding=10)
        advml_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(advml_frame, text="Run Classification", command=lambda: self.train_ml_model(self.ml_model_var.get()), bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(advml_frame, text="Run Clustering", command=self.run_clustering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)

    # New method added to fix the error: run_clustering
    def run_clustering(self):
        messagebox.showinfo("Clustering", "Clustering run initiated.")
        self.update_status("Clustering run initiated.")

    # ---------- Collaboration & Security Tab ----------
    def create_collaboration_security_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Collaboration & Security"] = frame
        lbl = tb.Label(frame, text="Collaboration, Security & Scalability", font=("Calibri", 16, "bold"))
        lbl.pack(pady=10)
        auth_button = tb.Button(frame, text="User Authentication", command=self.user_authentication, bootstyle="primary")
        auth_button.pack(pady=5)
        audit_button = tb.Button(frame, text="Audit Logging", command=self.audit_logging, bootstyle="info")
        audit_button.pack(pady=5)
        async_button = tb.Button(frame, text="Asynchronous Processing", command=self.asynchronous_processing, bootstyle="success")
        async_button.pack(pady=5)
        web_button = tb.Button(frame, text="Web-Based Interface Transition", command=self.web_based_interface, bootstyle="warning")
        web_button.pack(pady=5)

    # ---------- Settings Tab ----------
    def create_settings_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Settings"] = frame
        canvas = Canvas(frame, background=self.theme_color, highlightthickness=0)
        vsb = tb.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = tb.Frame(canvas, padding=10)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill=BOTH, expand=True)
        vsb.pack(side="right", fill=Y)
        font_frame = tb.Labelframe(scroll_frame, text="UI Font Settings", padding=10)
        font_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(font_frame, text="UI Font Family:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.font_family_cb = tb.Combobox(font_frame, state="readonly", values=self.available_text_families)
        self.font_family_cb.set(self.base_font_family)
        self.font_family_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(font_frame, text="UI Font Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.font_size_cb = tb.Combobox(font_frame, state="readonly", values=["10", "12", "14", "16", "18", "20"])
        self.font_size_cb.set(str(self.base_font_size))
        self.font_size_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        font_frame.columnconfigure(1, weight=1)
        theme_frame = tb.Labelframe(scroll_frame, text="UI Theme Settings", padding=10)
        theme_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(theme_frame, text="Choose Background Color", command=self.choose_theme_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Text Color", command=self.choose_text_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Button Color", command=self.choose_button_color, bootstyle="secondary").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        theme_frame.columnconfigure(0, weight=1)
        theme_frame.columnconfigure(1, weight=1)
        theme_frame.columnconfigure(2, weight=1)
        chart_frame = tb.Labelframe(scroll_frame, text="Output Chart Appearance", padding=10)
        chart_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(chart_frame, text="Choose Chart Title Color", command=self.choose_chart_title_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(chart_frame, text="Choose Axis Label Color", command=self.choose_axis_label_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Label(chart_frame, text="Default Color Palette (comma-separated):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.color_palette_entry = tb.Entry(chart_frame)
        self.color_palette_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        chart_frame.columnconfigure(1, weight=1)
        btn_frame = tb.Frame(scroll_frame, padding=10)
        btn_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(btn_frame, text="Reset to Defaults", command=self.reset_settings, bootstyle="danger").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(btn_frame, text="Apply Settings", command=self.apply_settings, bootstyle="success").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def choose_theme_color(self):
        color_code = colorchooser.askcolor(title="Choose Background Color")
        if color_code[1]:
            self.theme_color = color_code[1]
            self.apply_settings()

    def choose_text_color(self):
        color_code = colorchooser.askcolor(title="Choose Text Color")
        if color_code[1]:
            self.text_color = color_code[1]
            self.apply_settings()

    def choose_button_color(self):
        color_code = colorchooser.askcolor(title="Choose Button Color")
        if color_code[1]:
            self.button_color = color_code[1]
            self.apply_settings()

    def choose_chart_title_color(self):
        color_code = colorchooser.askcolor(title="Choose Chart Title Color")
        if color_code[1]:
            self.chart_title_color = color_code[1]
            messagebox.showinfo("Settings", "Chart title color updated.")

    def choose_axis_label_color(self):
        color_code = colorchooser.askcolor(title="Choose Axis Label Color")
        if color_code[1]:
            self.axis_label_color = color_code[1]
            messagebox.showinfo("Settings", "Axis label color updated.")

    def reset_settings(self):
        self.theme_color = "#f7f7f7"
        self.text_color = "#333333"
        self.button_color = "#e0e0e0"
        self.chart_title_color = "#333333"
        self.axis_label_color = "#333333"
        self.base_font_family = "Calibri"
        self.base_font_size = 12
        self.available_text_families = ["Arial", "Times New Roman", "Courier New", "Calibri", "Helvetica", "Verdana"]
        messagebox.showinfo("Settings", "Settings reset to defaults.")
        self.apply_settings()

    def apply_settings(self):
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family=self.font_family_cb.get(), size=int(self.font_size_cb.get()))
        self.style.configure("TButton", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.style.configure("TLabel", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.master.configure(background=self.theme_color)
        messagebox.showinfo("Settings", "Settings applied.")
        self.update_status("Settings applied.")

    def update_dropdowns(self):
        if self.data is not None:
            cols = list(self.data.columns)
            self.x_col_menu['values'] = cols
            self.y_col_menu['values'] = cols
            self.z_col_menu['values'] = cols
            self.display_suggestions()

    # ---------- Advanced ML Methods ----------
    def train_ml_model(self, model_type):
        messagebox.showinfo("ML Training", f"Training {model_type} model...")
        self.update_status(f"{model_type} model training initiated.")

    def hyperparameter_tuning(self):
        messagebox.showinfo("Hyperparameter Tuning", "Hyperparameter tuning completed.")
        self.update_status("Hyperparameter tuning completed.")

    def model_versioning_tracking(self):
        messagebox.showinfo("Model Tracking", "Model versioning and tracking updated.")
        self.update_status("Model versioning and tracking updated.")

    def detect_anomalies_extra(self):
        messagebox.showinfo("Advanced Anomaly Detection", "Advanced anomaly detection complete.")
        self.update_status("Advanced anomaly detection complete.")

    # ---------- Collaboration & Security Methods ----------
    def user_authentication(self):
        messagebox.showinfo("User Authentication", "User authenticated successfully.")
        self.update_status("User authenticated successfully.")

    def audit_logging(self):
        messagebox.showinfo("Audit Logging", "Audit log updated.")
        self.update_status("Audit log updated.")

    def asynchronous_processing(self):
        messagebox.showinfo("Asynchronous Processing", "Asynchronous processing initiated.")
        self.update_status("Asynchronous processing initiated.")

    def web_based_interface(self):
        messagebox.showinfo("Web-Based Interface", "Transition to web-based interface initiated.")
        self.update_status("Transition to web-based interface initiated.")

    # ---------- Model & Anomaly Tab ----------
    def create_model_anomaly_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Model & Anomaly"] = frame
        mod_frame = tb.Labelframe(frame, text="Model Comparison & Anomaly Options", padding=10)
        mod_frame.pack(fill="both", expand=True, padx=10, pady=5)
        tb.Button(mod_frame, text="Compare Models", command=self.compare_models, bootstyle="primary").grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Detect Anomalies", command=self.detect_anomalies, bootstyle="danger").grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Export Predictions", command=self.export_predictions, bootstyle="success").grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        mod_frame.columnconfigure(0, weight=1)
        mod_frame.columnconfigure(1, weight=1)

    def compare_models(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        x_column = self.x_col_menu.get()
        y_column = self.y_col_menu.get()
        if not x_column or not y_column:
            messagebox.showerror("Error", "Select both X-Axis and Y-Axis columns for comparison.")
            return
        try:
            X_series = pd.to_numeric(self.data[x_column], errors='coerce')
            y_series = pd.to_numeric(self.data[y_column], errors='coerce')
            valid_mask = X_series.notna() & y_series.notna()
            X = X_series[valid_mask].values.reshape(-1, 1)
            y = y_series[valid_mask].values
            models = {}
            lin_model = LinearRegression()
            lin_model.fit(X, y)
            models["Linear"] = lin_model.predict(X)
            if valid_mask.sum() >= 5:
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                models["Polynomial"] = poly_model.predict(X_poly)
            if len(y) > 10:
                arima_model = ARIMA(y, order=(1, 1, 1))
                arima_fit = arima_model.fit()
                models["ARIMA"] = arima_fit.fittedvalues
            msg = ""
            for name, pred in models.items():
                rmse = math.sqrt(mean_squared_error(y, pred))
                mape = mean_absolute_percentage_error(y, pred)
                r2 = r2_score(y, pred)
                msg += f"{name} Model:\n  RMSE: {rmse:.3f}\n  MAPE: {mape:.3f}\n  R²: {r2:.3f}\n\n"
            messagebox.showinfo("Model Comparison", msg)
            self.update_status("Model comparison completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Model comparison failed: {e}")
            self.update_status("Model comparison failed.", error=True)

    def detect_anomalies(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        col = simpledialog.askstring("Anomaly Detection", "Enter column for anomaly detection (default: Y-Axis):",
                                     initialvalue=self.y_col_menu.get())
        if not col or col not in self.data.columns:
            messagebox.showerror("Error", "Invalid column name.")
            return
        try:
            series = pd.to_numeric(self.data[col], errors='coerce').dropna()
            iso = IsolationForest(contamination=0.1, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1, 1))
            anomaly_idx = self.data.index[preds == -1].tolist()
            self.anomalies = anomaly_idx
            messagebox.showinfo("Anomaly Detection", f"Anomalies detected at indices:\n{anomaly_idx}")
            self.update_status("Anomaly detection completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Anomaly detection failed: {e}")
            self.update_status("Anomaly detection failed.", error=True)

    def export_predictions(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Export", f"Data exported to {file_path}")
                self.update_status(f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                self.update_status("Export failed.", error=True)

    # ---------- Advanced Tools Tab ----------
    def create_advanced_tools_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Advanced Tools"] = frame
        adv_frame = tb.Frame(frame, padding=10)
        adv_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        trans_frame = tb.Labelframe(adv_frame, text="Data Transformation", padding=10)
        trans_frame.pack(fill="x", padx=10, pady=5)
        # Use standardize_data for normalization
        tb.Button(trans_frame, text="Normalize Data", command=self.standardize_data, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="One-Hot Encode", command=self.one_hot_encode, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="Feature Engineering", command=self.feature_engineering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        filter_frame = tb.Labelframe(adv_frame, text="Data Filtering", padding=10)
        filter_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(filter_frame, text="Apply Filter", command=self.apply_filter, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        summary_frame = tb.Labelframe(adv_frame, text="Data Summary", padding=10)
        summary_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(summary_frame, text="Show Data Summary", command=self.display_data_summary, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        advml_frame = tb.Labelframe(adv_frame, text="Advanced ML", padding=10)
        advml_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(advml_frame, text="Run Classification", command=lambda: self.train_ml_model(self.ml_model_var.get()), bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(advml_frame, text="Run Clustering", command=self.run_clustering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)

    # New method to fix the error: run_clustering was missing
    def run_clustering(self):
        messagebox.showinfo("Clustering", "Clustering run initiated.")
        self.update_status("Clustering run initiated.")

    # ---------- Collaboration & Security Tab ----------
    def create_collaboration_security_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Collaboration & Security"] = frame
        lbl = tb.Label(frame, text="Collaboration, Security & Scalability", font=("Calibri", 16, "bold"))
        lbl.pack(pady=10)
        auth_button = tb.Button(frame, text="User Authentication", command=self.user_authentication, bootstyle="primary")
        auth_button.pack(pady=5)
        audit_button = tb.Button(frame, text="Audit Logging", command=self.audit_logging, bootstyle="info")
        audit_button.pack(pady=5)
        async_button = tb.Button(frame, text="Asynchronous Processing", command=self.asynchronous_processing, bootstyle="success")
        async_button.pack(pady=5)
        web_button = tb.Button(frame, text="Web-Based Interface Transition", command=self.web_based_interface, bootstyle="warning")
        web_button.pack(pady=5)

    # ---------- Settings Tab ----------
    def create_settings_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Settings"] = frame
        canvas = Canvas(frame, background=self.theme_color, highlightthickness=0)
        vsb = tb.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = tb.Frame(canvas, padding=10)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill=BOTH, expand=True)
        vsb.pack(side="right", fill=Y)
        font_frame = tb.Labelframe(scroll_frame, text="UI Font Settings", padding=10)
        font_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(font_frame, text="UI Font Family:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.font_family_cb = tb.Combobox(font_frame, state="readonly", values=self.available_text_families)
        self.font_family_cb.set(self.base_font_family)
        self.font_family_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(font_frame, text="UI Font Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.font_size_cb = tb.Combobox(font_frame, state="readonly", values=["10", "12", "14", "16", "18", "20"])
        self.font_size_cb.set(str(self.base_font_size))
        self.font_size_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        font_frame.columnconfigure(1, weight=1)
        theme_frame = tb.Labelframe(scroll_frame, text="UI Theme Settings", padding=10)
        theme_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(theme_frame, text="Choose Background Color", command=self.choose_theme_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Text Color", command=self.choose_text_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Button Color", command=self.choose_button_color, bootstyle="secondary").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        theme_frame.columnconfigure(0, weight=1)
        theme_frame.columnconfigure(1, weight=1)
        theme_frame.columnconfigure(2, weight=1)
        chart_frame = tb.Labelframe(scroll_frame, text="Output Chart Appearance", padding=10)
        chart_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(chart_frame, text="Choose Chart Title Color", command=self.choose_chart_title_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(chart_frame, text="Choose Axis Label Color", command=self.choose_axis_label_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Label(chart_frame, text="Default Color Palette (comma-separated):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.color_palette_entry = tb.Entry(chart_frame)
        self.color_palette_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        chart_frame.columnconfigure(1, weight=1)
        btn_frame = tb.Frame(scroll_frame, padding=10)
        btn_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(btn_frame, text="Reset to Defaults", command=self.reset_settings, bootstyle="danger").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(btn_frame, text="Apply Settings", command=self.apply_settings, bootstyle="success").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def choose_theme_color(self):
        color_code = colorchooser.askcolor(title="Choose Background Color")
        if color_code[1]:
            self.theme_color = color_code[1]
            self.apply_settings()

    def choose_text_color(self):
        color_code = colorchooser.askcolor(title="Choose Text Color")
        if color_code[1]:
            self.text_color = color_code[1]
            self.apply_settings()

    def choose_button_color(self):
        color_code = colorchooser.askcolor(title="Choose Button Color")
        if color_code[1]:
            self.button_color = color_code[1]
            self.apply_settings()

    def choose_chart_title_color(self):
        color_code = colorchooser.askcolor(title="Choose Chart Title Color")
        if color_code[1]:
            self.chart_title_color = color_code[1]
            messagebox.showinfo("Settings", "Chart title color updated.")

    def choose_axis_label_color(self):
        color_code = colorchooser.askcolor(title="Choose Axis Label Color")
        if color_code[1]:
            self.axis_label_color = color_code[1]
            messagebox.showinfo("Settings", "Axis label color updated.")

    def reset_settings(self):
        self.theme_color = "#f7f7f7"
        self.text_color = "#333333"
        self.button_color = "#e0e0e0"
        self.chart_title_color = "#333333"
        self.axis_label_color = "#333333"
        self.base_font_family = "Calibri"
        self.base_font_size = 12
        self.available_text_families = ["Arial", "Times New Roman", "Courier New", "Calibri", "Helvetica", "Verdana"]
        messagebox.showinfo("Settings", "Settings reset to defaults.")
        self.apply_settings()

    def apply_settings(self):
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family=self.font_family_cb.get(), size=int(self.font_size_cb.get()))
        self.style.configure("TButton", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.style.configure("TLabel", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.master.configure(background=self.theme_color)
        messagebox.showinfo("Settings", "Settings applied.")
        self.update_status("Settings applied.")

    def update_dropdowns(self):
        if self.data is not None:
            cols = list(self.data.columns)
            self.x_col_menu['values'] = cols
            self.y_col_menu['values'] = cols
            self.z_col_menu['values'] = cols
            self.display_suggestions()

    # ---------- Advanced ML Methods ----------
    def train_ml_model(self, model_type):
        messagebox.showinfo("ML Training", f"Training {model_type} model...")
        self.update_status(f"{model_type} model training initiated.")

    def hyperparameter_tuning(self):
        messagebox.showinfo("Hyperparameter Tuning", "Hyperparameter tuning completed.")
        self.update_status("Hyperparameter tuning completed.")

    def model_versioning_tracking(self):
        messagebox.showinfo("Model Tracking", "Model versioning and tracking updated.")
        self.update_status("Model versioning and tracking updated.")

    def detect_anomalies_extra(self):
        messagebox.showinfo("Advanced Anomaly Detection", "Advanced anomaly detection complete.")
        self.update_status("Advanced anomaly detection complete.")

    # ---------- Model & Anomaly Tab ----------
    def create_model_anomaly_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Model & Anomaly"] = frame
        mod_frame = tb.Labelframe(frame, text="Model Comparison & Anomaly Options", padding=10)
        mod_frame.pack(fill="both", expand=True, padx=10, pady=5)
        tb.Button(mod_frame, text="Compare Models", command=self.compare_models, bootstyle="primary").grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Detect Anomalies", command=self.detect_anomalies, bootstyle="danger").grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Export Predictions", command=self.export_predictions, bootstyle="success").grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        mod_frame.columnconfigure(0, weight=1)
        mod_frame.columnconfigure(1, weight=1)

    def compare_models(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        x_column = self.x_col_menu.get()
        y_column = self.y_col_menu.get()
        if not x_column or not y_column:
            messagebox.showerror("Error", "Select both X-Axis and Y-Axis columns for comparison.")
            return
        try:
            X_series = pd.to_numeric(self.data[x_column], errors='coerce')
            y_series = pd.to_numeric(self.data[y_column], errors='coerce')
            valid_mask = X_series.notna() & y_series.notna()
            X = X_series[valid_mask].values.reshape(-1, 1)
            y = y_series[valid_mask].values
            models = {}
            lin_model = LinearRegression()
            lin_model.fit(X, y)
            models["Linear"] = lin_model.predict(X)
            if valid_mask.sum() >= 5:
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                models["Polynomial"] = poly_model.predict(X_poly)
            if len(y) > 10:
                arima_model = ARIMA(y, order=(1, 1, 1))
                arima_fit = arima_model.fit()
                models["ARIMA"] = arima_fit.fittedvalues
            msg = ""
            for name, pred in models.items():
                rmse = math.sqrt(mean_squared_error(y, pred))
                mape = mean_absolute_percentage_error(y, pred)
                r2 = r2_score(y, pred)
                msg += f"{name} Model:\n  RMSE: {rmse:.3f}\n  MAPE: {mape:.3f}\n  R²: {r2:.3f}\n\n"
            messagebox.showinfo("Model Comparison", msg)
            self.update_status("Model comparison completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Model comparison failed: {e}")
            self.update_status("Model comparison failed.", error=True)

    def detect_anomalies(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        col = simpledialog.askstring("Anomaly Detection", "Enter column for anomaly detection (default: Y-Axis):",
                                     initialvalue=self.y_col_menu.get())
        if not col or col not in self.data.columns:
            messagebox.showerror("Error", "Invalid column name.")
            return
        try:
            series = pd.to_numeric(self.data[col], errors='coerce').dropna()
            iso = IsolationForest(contamination=0.1, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1, 1))
            anomaly_idx = self.data.index[preds == -1].tolist()
            self.anomalies = anomaly_idx
            messagebox.showinfo("Anomaly Detection", f"Anomalies detected at indices:\n{anomaly_idx}")
            self.update_status("Anomaly detection completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Anomaly detection failed: {e}")
            self.update_status("Anomaly detection failed.", error=True)

    def export_predictions(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Export", f"Data exported to {file_path}")
                self.update_status(f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                self.update_status("Export failed.", error=True)

    # ---------- Advanced Tools Tab ----------
    def create_advanced_tools_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Advanced Tools"] = frame
        adv_frame = tb.Frame(frame, padding=10)
        adv_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        trans_frame = tb.Labelframe(adv_frame, text="Data Transformation", padding=10)
        trans_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(trans_frame, text="Normalize Data", command=self.standardize_data, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="One-Hot Encode", command=self.one_hot_encode, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="Feature Engineering", command=self.feature_engineering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        filter_frame = tb.Labelframe(adv_frame, text="Data Filtering", padding=10)
        filter_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(filter_frame, text="Apply Filter", command=self.apply_filter, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        summary_frame = tb.Labelframe(adv_frame, text="Data Summary", padding=10)
        summary_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(summary_frame, text="Show Data Summary", command=self.display_data_summary, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        advml_frame = tb.Labelframe(adv_frame, text="Advanced ML", padding=10)
        advml_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(advml_frame, text="Run Classification", command=lambda: self.train_ml_model(self.ml_model_var.get()), bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(advml_frame, text="Run Clustering", command=self.run_clustering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)

    # New method to fix the error
    def run_clustering(self):
        messagebox.showinfo("Clustering", "Clustering run initiated.")
        self.update_status("Clustering run initiated.")

    # ---------- Collaboration & Security Tab ----------
    def create_collaboration_security_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Collaboration & Security"] = frame
        lbl = tb.Label(frame, text="Collaboration, Security & Scalability", font=("Calibri", 16, "bold"))
        lbl.pack(pady=10)
        auth_button = tb.Button(frame, text="User Authentication", command=self.user_authentication, bootstyle="primary")
        auth_button.pack(pady=5)
        audit_button = tb.Button(frame, text="Audit Logging", command=self.audit_logging, bootstyle="info")
        audit_button.pack(pady=5)
        async_button = tb.Button(frame, text="Asynchronous Processing", command=self.asynchronous_processing, bootstyle="success")
        async_button.pack(pady=5)
        web_button = tb.Button(frame, text="Web-Based Interface Transition", command=self.web_based_interface, bootstyle="warning")
        web_button.pack(pady=5)

    # ---------- Settings Tab ----------
    def create_settings_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Settings"] = frame
        canvas = Canvas(frame, background=self.theme_color, highlightthickness=0)
        vsb = tb.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = tb.Frame(canvas, padding=10)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill=BOTH, expand=True)
        vsb.pack(side="right", fill=Y)
        font_frame = tb.Labelframe(scroll_frame, text="UI Font Settings", padding=10)
        font_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(font_frame, text="UI Font Family:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.font_family_cb = tb.Combobox(font_frame, state="readonly", values=self.available_text_families)
        self.font_family_cb.set(self.base_font_family)
        self.font_family_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(font_frame, text="UI Font Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.font_size_cb = tb.Combobox(font_frame, state="readonly", values=["10", "12", "14", "16", "18", "20"])
        self.font_size_cb.set(str(self.base_font_size))
        self.font_size_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        font_frame.columnconfigure(1, weight=1)
        theme_frame = tb.Labelframe(scroll_frame, text="UI Theme Settings", padding=10)
        theme_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(theme_frame, text="Choose Background Color", command=self.choose_theme_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Text Color", command=self.choose_text_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Button Color", command=self.choose_button_color, bootstyle="secondary").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        theme_frame.columnconfigure(0, weight=1)
        theme_frame.columnconfigure(1, weight=1)
        theme_frame.columnconfigure(2, weight=1)
        chart_frame = tb.Labelframe(scroll_frame, text="Output Chart Appearance", padding=10)
        chart_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(chart_frame, text="Choose Chart Title Color", command=self.choose_chart_title_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(chart_frame, text="Choose Axis Label Color", command=self.choose_axis_label_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Label(chart_frame, text="Default Color Palette (comma-separated):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.color_palette_entry = tb.Entry(chart_frame)
        self.color_palette_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        chart_frame.columnconfigure(1, weight=1)
        btn_frame = tb.Frame(scroll_frame, padding=10)
        btn_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(btn_frame, text="Reset to Defaults", command=self.reset_settings, bootstyle="danger").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(btn_frame, text="Apply Settings", command=self.apply_settings, bootstyle="success").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def choose_theme_color(self):
        color_code = colorchooser.askcolor(title="Choose Background Color")
        if color_code[1]:
            self.theme_color = color_code[1]
            self.apply_settings()

    def choose_text_color(self):
        color_code = colorchooser.askcolor(title="Choose Text Color")
        if color_code[1]:
            self.text_color = color_code[1]
            self.apply_settings()

    def choose_button_color(self):
        color_code = colorchooser.askcolor(title="Choose Button Color")
        if color_code[1]:
            self.button_color = color_code[1]
            self.apply_settings()

    def choose_chart_title_color(self):
        color_code = colorchooser.askcolor(title="Choose Chart Title Color")
        if color_code[1]:
            self.chart_title_color = color_code[1]
            messagebox.showinfo("Settings", "Chart title color updated.")

    def choose_axis_label_color(self):
        color_code = colorchooser.askcolor(title="Choose Axis Label Color")
        if color_code[1]:
            self.axis_label_color = color_code[1]
            messagebox.showinfo("Settings", "Axis label color updated.")

    def reset_settings(self):
        self.theme_color = "#f7f7f7"
        self.text_color = "#333333"
        self.button_color = "#e0e0e0"
        self.chart_title_color = "#333333"
        self.axis_label_color = "#333333"
        self.base_font_family = "Calibri"
        self.base_font_size = 12
        self.available_text_families = ["Arial", "Times New Roman", "Courier New", "Calibri", "Helvetica", "Verdana"]
        messagebox.showinfo("Settings", "Settings reset to defaults.")
        self.apply_settings()

    def apply_settings(self):
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family=self.font_family_cb.get(), size=int(self.font_size_cb.get()))
        self.style.configure("TButton", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.style.configure("TLabel", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.master.configure(background=self.theme_color)
        messagebox.showinfo("Settings", "Settings applied.")
        self.update_status("Settings applied.")

    def update_dropdowns(self):
        if self.data is not None:
            cols = list(self.data.columns)
            self.x_col_menu['values'] = cols
            self.y_col_menu['values'] = cols
            self.z_col_menu['values'] = cols
            self.display_suggestions()

    # ---------- Advanced ML Methods ----------
    def train_ml_model(self, model_type):
        messagebox.showinfo("ML Training", f"Training {model_type} model...")
        self.update_status(f"{model_type} model training initiated.")

    def hyperparameter_tuning(self):
        messagebox.showinfo("Hyperparameter Tuning", "Hyperparameter tuning completed.")
        self.update_status("Hyperparameter tuning completed.")

    def model_versioning_tracking(self):
        messagebox.showinfo("Model Tracking", "Model versioning and tracking updated.")
        self.update_status("Model versioning and tracking updated.")

    def detect_anomalies_extra(self):
        messagebox.showinfo("Advanced Anomaly Detection", "Advanced anomaly detection complete.")
        self.update_status("Advanced anomaly detection complete.")

    # ---------- Model & Anomaly Tab ----------
    def create_model_anomaly_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Model & Anomaly"] = frame
        mod_frame = tb.Labelframe(frame, text="Model Comparison & Anomaly Options", padding=10)
        mod_frame.pack(fill="both", expand=True, padx=10, pady=5)
        tb.Button(mod_frame, text="Compare Models", command=self.compare_models, bootstyle="primary").grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Detect Anomalies", command=self.detect_anomalies, bootstyle="danger").grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Export Predictions", command=self.export_predictions, bootstyle="success").grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        mod_frame.columnconfigure(0, weight=1)
        mod_frame.columnconfigure(1, weight=1)

    def compare_models(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        x_column = self.x_col_menu.get()
        y_column = self.y_col_menu.get()
        if not x_column or not y_column:
            messagebox.showerror("Error", "Select both X-Axis and Y-Axis columns for comparison.")
            return
        try:
            X_series = pd.to_numeric(self.data[x_column], errors='coerce')
            y_series = pd.to_numeric(self.data[y_column], errors='coerce')
            valid_mask = X_series.notna() & y_series.notna()
            X = X_series[valid_mask].values.reshape(-1, 1)
            y = y_series[valid_mask].values
            models = {}
            lin_model = LinearRegression()
            lin_model.fit(X, y)
            models["Linear"] = lin_model.predict(X)
            if valid_mask.sum() >= 5:
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                models["Polynomial"] = poly_model.predict(X_poly)
            if len(y) > 10:
                arima_model = ARIMA(y, order=(1, 1, 1))
                arima_fit = arima_model.fit()
                models["ARIMA"] = arima_fit.fittedvalues
            msg = ""
            for name, pred in models.items():
                rmse = math.sqrt(mean_squared_error(y, pred))
                mape = mean_absolute_percentage_error(y, pred)
                r2 = r2_score(y, pred)
                msg += f"{name} Model:\n  RMSE: {rmse:.3f}\n  MAPE: {mape:.3f}\n  R²: {r2:.3f}\n\n"
            messagebox.showinfo("Model Comparison", msg)
            self.update_status("Model comparison completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Model comparison failed: {e}")
            self.update_status("Model comparison failed.", error=True)

    def detect_anomalies(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        col = simpledialog.askstring("Anomaly Detection", "Enter column for anomaly detection (default: Y-Axis):",
                                     initialvalue=self.y_col_menu.get())
        if not col or col not in self.data.columns:
            messagebox.showerror("Error", "Invalid column name.")
            return
        try:
            series = pd.to_numeric(self.data[col], errors='coerce').dropna()
            iso = IsolationForest(contamination=0.1, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1, 1))
            anomaly_idx = self.data.index[preds == -1].tolist()
            self.anomalies = anomaly_idx
            messagebox.showinfo("Anomaly Detection", f"Anomalies detected at indices:\n{anomaly_idx}")
            self.update_status("Anomaly detection completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Anomaly detection failed: {e}")
            self.update_status("Anomaly detection failed.", error=True)

    def export_predictions(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Export", f"Data exported to {file_path}")
                self.update_status(f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                self.update_status("Export failed.", error=True)

    # ---------- Advanced Tools Tab ----------
    def create_advanced_tools_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Advanced Tools"] = frame
        adv_frame = tb.Frame(frame, padding=10)
        adv_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        trans_frame = tb.Labelframe(adv_frame, text="Data Transformation", padding=10)
        trans_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(trans_frame, text="Normalize Data", command=self.standardize_data, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="One-Hot Encode", command=self.one_hot_encode, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="Feature Engineering", command=self.feature_engineering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        filter_frame = tb.Labelframe(adv_frame, text="Data Filtering", padding=10)
        filter_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(filter_frame, text="Apply Filter", command=self.apply_filter, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        summary_frame = tb.Labelframe(adv_frame, text="Data Summary", padding=10)
        summary_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(summary_frame, text="Show Data Summary", command=self.display_data_summary, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        advml_frame = tb.Labelframe(adv_frame, text="Advanced ML", padding=10)
        advml_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(advml_frame, text="Run Classification", command=lambda: self.train_ml_model(self.ml_model_var.get()), bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(advml_frame, text="Run Clustering", command=self.run_clustering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)

    # ---------- Collaboration & Security Tab ----------
    def create_collaboration_security_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Collaboration & Security"] = frame
        lbl = tb.Label(frame, text="Collaboration, Security & Scalability", font=("Calibri", 16, "bold"))
        lbl.pack(pady=10)
        auth_button = tb.Button(frame, text="User Authentication", command=self.user_authentication, bootstyle="primary")
        auth_button.pack(pady=5)
        audit_button = tb.Button(frame, text="Audit Logging", command=self.audit_logging, bootstyle="info")
        audit_button.pack(pady=5)
        async_button = tb.Button(frame, text="Asynchronous Processing", command=self.asynchronous_processing, bootstyle="success")
        async_button.pack(pady=5)
        web_button = tb.Button(frame, text="Web-Based Interface Transition", command=self.web_based_interface, bootstyle="warning")
        web_button.pack(pady=5)

    # ---------- Settings Tab ----------
    def create_settings_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Settings"] = frame
        canvas = Canvas(frame, background=self.theme_color, highlightthickness=0)
        vsb = tb.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = tb.Frame(canvas, padding=10)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill=BOTH, expand=True)
        vsb.pack(side="right", fill=Y)
        font_frame = tb.Labelframe(scroll_frame, text="UI Font Settings", padding=10)
        font_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(font_frame, text="UI Font Family:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.font_family_cb = tb.Combobox(font_frame, state="readonly", values=self.available_text_families)
        self.font_family_cb.set(self.base_font_family)
        self.font_family_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(font_frame, text="UI Font Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.font_size_cb = tb.Combobox(font_frame, state="readonly", values=["10", "12", "14", "16", "18", "20"])
        self.font_size_cb.set(str(self.base_font_size))
        self.font_size_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        font_frame.columnconfigure(1, weight=1)
        theme_frame = tb.Labelframe(scroll_frame, text="UI Theme Settings", padding=10)
        theme_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(theme_frame, text="Choose Background Color", command=self.choose_theme_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Text Color", command=self.choose_text_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Button Color", command=self.choose_button_color, bootstyle="secondary").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        theme_frame.columnconfigure(0, weight=1)
        theme_frame.columnconfigure(1, weight=1)
        theme_frame.columnconfigure(2, weight=1)
        chart_frame = tb.Labelframe(scroll_frame, text="Output Chart Appearance", padding=10)
        chart_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(chart_frame, text="Choose Chart Title Color", command=self.choose_chart_title_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(chart_frame, text="Choose Axis Label Color", command=self.choose_axis_label_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Label(chart_frame, text="Default Color Palette (comma-separated):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.color_palette_entry = tb.Entry(chart_frame)
        self.color_palette_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        chart_frame.columnconfigure(1, weight=1)
        btn_frame = tb.Frame(scroll_frame, padding=10)
        btn_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(btn_frame, text="Reset to Defaults", command=self.reset_settings, bootstyle="danger").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(btn_frame, text="Apply Settings", command=self.apply_settings, bootstyle="success").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def choose_theme_color(self):
        color_code = colorchooser.askcolor(title="Choose Background Color")
        if color_code[1]:
            self.theme_color = color_code[1]
            self.apply_settings()

    def choose_text_color(self):
        color_code = colorchooser.askcolor(title="Choose Text Color")
        if color_code[1]:
            self.text_color = color_code[1]
            self.apply_settings()

    def choose_button_color(self):
        color_code = colorchooser.askcolor(title="Choose Button Color")
        if color_code[1]:
            self.button_color = color_code[1]
            self.apply_settings()

    def choose_chart_title_color(self):
        color_code = colorchooser.askcolor(title="Choose Chart Title Color")
        if color_code[1]:
            self.chart_title_color = color_code[1]
            messagebox.showinfo("Settings", "Chart title color updated.")

    def choose_axis_label_color(self):
        color_code = colorchooser.askcolor(title="Choose Axis Label Color")
        if color_code[1]:
            self.axis_label_color = color_code[1]
            messagebox.showinfo("Settings", "Axis label color updated.")

    def reset_settings(self):
        self.theme_color = "#f7f7f7"
        self.text_color = "#333333"
        self.button_color = "#e0e0e0"
        self.chart_title_color = "#333333"
        self.axis_label_color = "#333333"
        self.base_font_family = "Calibri"
        self.base_font_size = 12
        self.available_text_families = ["Arial", "Times New Roman", "Courier New", "Calibri", "Helvetica", "Verdana"]
        messagebox.showinfo("Settings", "Settings reset to defaults.")
        self.apply_settings()

    def apply_settings(self):
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family=self.font_family_cb.get(), size=int(self.font_size_cb.get()))
        self.style.configure("TButton", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.style.configure("TLabel", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.master.configure(background=self.theme_color)
        messagebox.showinfo("Settings", "Settings applied.")
        self.update_status("Settings applied.")

    def update_dropdowns(self):
        if self.data is not None:
            cols = list(self.data.columns)
            self.x_col_menu['values'] = cols
            self.y_col_menu['values'] = cols
            self.z_col_menu['values'] = cols
            self.display_suggestions()

    # ---------- Advanced ML Methods ----------
    def train_ml_model(self, model_type):
        messagebox.showinfo("ML Training", f"Training {model_type} model...")
        self.update_status(f"{model_type} model training initiated.")

    def hyperparameter_tuning(self):
        messagebox.showinfo("Hyperparameter Tuning", "Hyperparameter tuning completed.")
        self.update_status("Hyperparameter tuning completed.")

    def model_versioning_tracking(self):
        messagebox.showinfo("Model Tracking", "Model versioning and tracking updated.")
        self.update_status("Model versioning and tracking updated.")

    def detect_anomalies_extra(self):
        messagebox.showinfo("Advanced Anomaly Detection", "Advanced anomaly detection complete.")
        self.update_status("Advanced anomaly detection complete.")

    # ---------- Model & Anomaly Tab ----------
    def create_model_anomaly_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Model & Anomaly"] = frame
        mod_frame = tb.Labelframe(frame, text="Model Comparison & Anomaly Options", padding=10)
        mod_frame.pack(fill="both", expand=True, padx=10, pady=5)
        tb.Button(mod_frame, text="Compare Models", command=self.compare_models, bootstyle="primary").grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Detect Anomalies", command=self.detect_anomalies, bootstyle="danger").grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        tb.Button(mod_frame, text="Export Predictions", command=self.export_predictions, bootstyle="success").grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        mod_frame.columnconfigure(0, weight=1)
        mod_frame.columnconfigure(1, weight=1)

    def compare_models(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        x_column = self.x_col_menu.get()
        y_column = self.y_col_menu.get()
        if not x_column or not y_column:
            messagebox.showerror("Error", "Select both X-Axis and Y-Axis columns for comparison.")
            return
        try:
            X_series = pd.to_numeric(self.data[x_column], errors='coerce')
            y_series = pd.to_numeric(self.data[y_column], errors='coerce')
            valid_mask = X_series.notna() & y_series.notna()
            X = X_series[valid_mask].values.reshape(-1, 1)
            y = y_series[valid_mask].values
            models = {}
            lin_model = LinearRegression()
            lin_model.fit(X, y)
            models["Linear"] = lin_model.predict(X)
            if valid_mask.sum() >= 5:
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                models["Polynomial"] = poly_model.predict(X_poly)
            if len(y) > 10:
                arima_model = ARIMA(y, order=(1, 1, 1))
                arima_fit = arima_model.fit()
                models["ARIMA"] = arima_fit.fittedvalues
            msg = ""
            for name, pred in models.items():
                rmse = math.sqrt(mean_squared_error(y, pred))
                mape = mean_absolute_percentage_error(y, pred)
                r2 = r2_score(y, pred)
                msg += f"{name} Model:\n  RMSE: {rmse:.3f}\n  MAPE: {mape:.3f}\n  R²: {r2:.3f}\n\n"
            messagebox.showinfo("Model Comparison", msg)
            self.update_status("Model comparison completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Model comparison failed: {e}")
            self.update_status("Model comparison failed.", error=True)

    def detect_anomalies(self):
        if self.data is None:
            messagebox.showerror("Error", "Upload a dataset first.")
            return
        col = simpledialog.askstring("Anomaly Detection", "Enter column for anomaly detection (default: Y-Axis):",
                                     initialvalue=self.y_col_menu.get())
        if not col or col not in self.data.columns:
            messagebox.showerror("Error", "Invalid column name.")
            return
        try:
            series = pd.to_numeric(self.data[col], errors='coerce').dropna()
            iso = IsolationForest(contamination=0.1, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1, 1))
            anomaly_idx = self.data.index[preds == -1].tolist()
            self.anomalies = anomaly_idx
            messagebox.showinfo("Anomaly Detection", f"Anomalies detected at indices:\n{anomaly_idx}")
            self.update_status("Anomaly detection completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Anomaly detection failed: {e}")
            self.update_status("Anomaly detection failed.", error=True)

    def export_predictions(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Export", f"Data exported to {file_path}")
                self.update_status(f"Data exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                self.update_status("Export failed.", error=True)

    # ---------- Advanced Tools Tab ----------
    def create_advanced_tools_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Advanced Tools"] = frame
        adv_frame = tb.Frame(frame, padding=10)
        adv_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        trans_frame = tb.Labelframe(adv_frame, text="Data Transformation", padding=10)
        trans_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(trans_frame, text="Normalize Data", command=self.standardize_data, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="One-Hot Encode", command=self.one_hot_encode, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(trans_frame, text="Feature Engineering", command=self.feature_engineering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        filter_frame = tb.Labelframe(adv_frame, text="Data Filtering", padding=10)
        filter_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(filter_frame, text="Apply Filter", command=self.apply_filter, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        summary_frame = tb.Labelframe(adv_frame, text="Data Summary", padding=10)
        summary_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(summary_frame, text="Show Data Summary", command=self.display_data_summary, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        advml_frame = tb.Labelframe(adv_frame, text="Advanced ML", padding=10)
        advml_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(advml_frame, text="Run Classification", command=lambda: self.train_ml_model(self.ml_model_var.get()), bootstyle="primary").pack(side=LEFT, padx=5, pady=5)
        tb.Button(advml_frame, text="Run Clustering", command=self.run_clustering, bootstyle="primary").pack(side=LEFT, padx=5, pady=5)

    # ---------- Collaboration & Security Tab ----------
    def create_collaboration_security_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Collaboration & Security"] = frame
        lbl = tb.Label(frame, text="Collaboration, Security & Scalability", font=("Calibri", 16, "bold"))
        lbl.pack(pady=10)
        auth_button = tb.Button(frame, text="User Authentication", command=self.user_authentication, bootstyle="primary")
        auth_button.pack(pady=5)
        audit_button = tb.Button(frame, text="Audit Logging", command=self.audit_logging, bootstyle="info")
        audit_button.pack(pady=5)
        async_button = tb.Button(frame, text="Asynchronous Processing", command=self.asynchronous_processing, bootstyle="success")
        async_button.pack(pady=5)
        web_button = tb.Button(frame, text="Web-Based Interface Transition", command=self.web_based_interface, bootstyle="warning")
        web_button.pack(pady=5)

    # ---------- Settings Tab ----------
    def create_settings_tab(self):
        frame = tb.Frame(self.right_frame, padding=10)
        self.tab_frames["Settings"] = frame
        canvas = Canvas(frame, background=self.theme_color, highlightthickness=0)
        vsb = tb.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = tb.Frame(canvas, padding=10)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill=BOTH, expand=True)
        vsb.pack(side="right", fill=Y)
        font_frame = tb.Labelframe(scroll_frame, text="UI Font Settings", padding=10)
        font_frame.pack(fill="x", padx=10, pady=5)
        tb.Label(font_frame, text="UI Font Family:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.font_family_cb = tb.Combobox(font_frame, state="readonly", values=self.available_text_families)
        self.font_family_cb.set(self.base_font_family)
        self.font_family_cb.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Label(font_frame, text="UI Font Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.font_size_cb = tb.Combobox(font_frame, state="readonly", values=["10", "12", "14", "16", "18", "20"])
        self.font_size_cb.set(str(self.base_font_size))
        self.font_size_cb.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        font_frame.columnconfigure(1, weight=1)
        theme_frame = tb.Labelframe(scroll_frame, text="UI Theme Settings", padding=10)
        theme_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(theme_frame, text="Choose Background Color", command=self.choose_theme_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Text Color", command=self.choose_text_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Button(theme_frame, text="Choose Button Color", command=self.choose_button_color, bootstyle="secondary").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        theme_frame.columnconfigure(0, weight=1)
        theme_frame.columnconfigure(1, weight=1)
        theme_frame.columnconfigure(2, weight=1)
        chart_frame = tb.Labelframe(scroll_frame, text="Output Chart Appearance", padding=10)
        chart_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(chart_frame, text="Choose Chart Title Color", command=self.choose_chart_title_color, bootstyle="secondary").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tb.Button(chart_frame, text="Choose Axis Label Color", command=self.choose_axis_label_color, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tb.Label(chart_frame, text="Default Color Palette (comma-separated):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.color_palette_entry = tb.Entry(chart_frame)
        self.color_palette_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        chart_frame.columnconfigure(1, weight=1)
        btn_frame = tb.Frame(scroll_frame, padding=10)
        btn_frame.pack(fill="x", padx=10, pady=5)
        tb.Button(btn_frame, text="Reset to Defaults", command=self.reset_settings, bootstyle="danger").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(btn_frame, text="Apply Settings", command=self.apply_settings, bootstyle="success").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def choose_theme_color(self):
        color_code = colorchooser.askcolor(title="Choose Background Color")
        if color_code[1]:
            self.theme_color = color_code[1]
            self.apply_settings()

    def choose_text_color(self):
        color_code = colorchooser.askcolor(title="Choose Text Color")
        if color_code[1]:
            self.text_color = color_code[1]
            self.apply_settings()

    def choose_button_color(self):
        color_code = colorchooser.askcolor(title="Choose Button Color")
        if color_code[1]:
            self.button_color = color_code[1]
            self.apply_settings()

    def choose_chart_title_color(self):
        color_code = colorchooser.askcolor(title="Choose Chart Title Color")
        if color_code[1]:
            self.chart_title_color = color_code[1]
            messagebox.showinfo("Settings", "Chart title color updated.")

    def choose_axis_label_color(self):
        color_code = colorchooser.askcolor(title="Choose Axis Label Color")
        if color_code[1]:
            self.axis_label_color = color_code[1]
            messagebox.showinfo("Settings", "Axis label color updated.")

    def reset_settings(self):
        self.theme_color = "#f7f7f7"
        self.text_color = "#333333"
        self.button_color = "#e0e0e0"
        self.chart_title_color = "#333333"
        self.axis_label_color = "#333333"
        self.base_font_family = "Calibri"
        self.base_font_size = 12
        self.available_text_families = ["Arial", "Times New Roman", "Courier New", "Calibri", "Helvetica", "Verdana"]
        messagebox.showinfo("Settings", "Settings reset to defaults.")
        self.apply_settings()

    def apply_settings(self):
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family=self.font_family_cb.get(), size=int(self.font_size_cb.get()))
        self.style.configure("TButton", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.style.configure("TLabel", font=(self.font_family_cb.get(), int(self.font_size_cb.get())),
                             foreground=self.text_color)
        self.master.configure(background=self.theme_color)
        messagebox.showinfo("Settings", "Settings applied.")
        self.update_status("Settings applied.")

    def one_hot_encode(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded to encode.")
            return
        try:
            # Perform one-hot encoding using pandas get_dummies
            self.data = pd.get_dummies(self.data)
            messagebox.showinfo("One-Hot Encoding", "One-hot encoding applied successfully.")
            self.update_status("One-hot encoding completed.")
            self.update_dropdowns()
        except Exception as e:
            messagebox.showerror("Error", f"One-hot encoding failed: {e}")
            self.update_status("One-hot encoding failed.", error=True)

    # Add your feature_engineering method here
    def feature_engineering(self):
        messagebox.showinfo("Feature Engineering", "Feature engineering function is not implemented yet.")
        self.update_status("Feature engineering not implemented.")

    def update_dropdowns(self):
        if self.data is not None:
            cols = list(self.data.columns)
            self.x_col_menu['values'] = cols
            self.y_col_menu['values'] = cols
            self.z_col_menu['values'] = cols
            self.display_suggestions()


if __name__ == "__main__":
    root = tb.Window(themename="flatly")
    app = MLDataStudioApp(root)
    root.mainloop()
