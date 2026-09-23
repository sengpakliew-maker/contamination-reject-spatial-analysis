"""All Flet UI construction, controllers, callbacks, and application workflow."""
from __future__ import annotations
import asyncio
import base64
import json
from pathlib import Path
import flet as ft
import pandas as pd
from analysis import create_method1_plot, create_method2_plot, plot_device_map, run_method1, run_method2
from data import (
    add_effective_unit,
    add_strip_anomaly_feature,
    apply_anomaly_preprocessing,
    batch_lot_records,
    build_batch_overview,
    data_cleaning,
    get_analysis_population as get_analysis_population_core,
    load_excel_data,
    prepare_device_analysis_data,
    selected_batch_keys,
)
from output import (
    create_device_plot,
    export_method1_excel,
    export_method2_excel,
    plot_control_from_bytes,
    show_method_results,
)
DEFAULT_DEVICE_CONFIG = {'VEGA': {'rows': 15, 'cols': 64, 'panel_size': 16, 'criteria': 'A'}, 'SOLARIS-XRF3': {'rows': 17, 'cols': 76, 'panel_size': 19, 'criteria': 'A'}, 'SQUID-ICD82PRV': {'rows': 17, 'cols': 72, 'panel_size': 18, 'criteria': 'B'}}
DEVICE_CONFIG_FILE = Path(__file__).resolve().parent / 'device_config.json'

def _plot_control_from_bytes(image_bytes, width=1100, height=None):
    """Create a Flet image control from PNG bytes using the application UI context."""
    return plot_control_from_bytes(ft, image_bytes, width, height)

def configure_page(page):
    """Apply the application's global page configuration."""
    page.title = 'Process Analysis Tool'
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO

def create_app_controls(page):
    """Create the application's static controls and return them by name."""
    title = ft.Text('Process Analysis Tool', size=30, weight=ft.FontWeight.BOLD)
    upload_status = ft.Text('No file selected.', size=14)
    selected_file_text = ft.Text('', size=14)
    device_checkboxes = ft.Column(spacing=5, scroll=ft.ScrollMode.AUTO)
    start_date = ft.TextField(label='Start date', width=250, value='2026-03-27')
    end_date = ft.TextField(label='End date', width=250, value='2026-07-27')
    data_selection_mode = ft.Dropdown(label='Data selection', width=220, options=[ft.DropdownOption('Date Range'), ft.DropdownOption('Batch Helper')], value='Date Range')
    batch_status = ft.Text('', size=14)
    method2_reference_list = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO)
    method2_current_list = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO)
    method2_clear_a_button = ft.Button(content='Clear A')
    method2_clear_b_button = ft.Button(content='Clear B')
    method2_reference_column = ft.Column([ft.Text('Reference A', size=15, weight=ft.FontWeight.BOLD), method2_reference_list, method2_clear_a_button], spacing=5)
    method2_current_column = ft.Column([ft.Text('Current B', size=15, weight=ft.FontWeight.BOLD), method2_current_list, method2_clear_b_button], spacing=5)
    method2_batch_assignment = ft.Container(content=ft.Row([ft.Container(content=method2_reference_column, width=420, padding=8, border=ft.Border.all(1)), ft.Container(content=method2_current_column, width=420, padding=8, border=ft.Border.all(1))], spacing=10), visible=False)
    batch_table = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO)
    batch_find_button = ft.Button(content='Find Batches', icon=ft.Icons.SEARCH)
    batch_select_all_button = ft.Button(content='Select All')
    batch_clear_all_button = ft.Button(content='Clear All')
    batch_use_button = ft.Button(content='Use Selected Batches')
    batch_hide_button = ft.Button(content='Hide')
    batch_helper_header = ft.Row([ft.Text('Batch Helper', size=18, weight=ft.FontWeight.BOLD), batch_find_button, batch_select_all_button, batch_clear_all_button, batch_use_button, batch_hide_button], spacing=8, wrap=True)
    batch_helper_container = ft.Container(content=ft.Column([batch_helper_header, batch_status, method2_batch_assignment, batch_table], spacing=6), padding=10, border=ft.Border.all(1), visible=False)
    batch_show_button = ft.Button(content='Show Batch Helper', visible=False)
    analysis_mode = ft.Dropdown(label='Analysis mode', width=300, options=[ft.DropdownOption('Individual Strip'), ft.DropdownOption('N-Day Aggregation'), ft.DropdownOption('Weekly'), ft.DropdownOption('Monthly'), ft.DropdownOption('Method 1 — DBSCAN Clustering'), ft.DropdownOption('Method 2 — Batch Comparison')], value='Individual Strip')
    n_day_field = ft.TextField(label='Number of days', width=180, hint_text='e.g. 7', visible=False, keyboard_type=ft.KeyboardType.NUMBER)
    METHOD1_EPS = 2.24
    METHOD1_MIN_SAMPLES = 3
    METHOD1_MIN_CLUSTER_SIZE = 3
    method1_top_strip_dropdown = ft.Dropdown(label='Strips to plot', width=170, options=[ft.DropdownOption('Top 1'), ft.DropdownOption('Top 3'), ft.DropdownOption('Top 5'), ft.DropdownOption('Top 10'), ft.DropdownOption('Top 20'), ft.DropdownOption('All')], value='Top 10', visible=False)
    method_top_bins_field = ft.TextField(label='Top bins', width=130, value='10', keyboard_type=ft.KeyboardType.NUMBER, visible=False)
    method_n_iter_field = ft.TextField(label='Iterations', width=130, value='500', keyboard_type=ft.KeyboardType.NUMBER, visible=False)
    method_anomaly_checkbox = ft.Checkbox(label='Strip-level anomaly filter', value=False, visible=False)
    method_anomaly_z_field = ft.TextField(label='Z-score', width=130, value='3', keyboard_type=ft.KeyboardType.NUMBER, visible=False)
    method2_reference_start = ft.TextField(label='Reference start', width=170, value='2026-03-27', visible=False)
    method2_reference_end = ft.TextField(label='Reference end', width=170, value='2026-04-26', visible=False)
    method2_current_start = ft.TextField(label='Current start', width=170, value='2026-04-27', visible=False)
    method2_current_end = ft.TextField(label='Current end', width=170, value='2026-05-26', visible=False)
    method2_p_threshold = ft.TextField(label='p threshold %', width=140, value='5', keyboard_type=ft.KeyboardType.NUMBER, visible=False)
    method_export_button = ft.Button(content='Export Method Excel', visible=False)
    method_status = ft.Text('', size=14)
    global_loading_ring = ft.ProgressRing(width=20, height=20)
    global_loading_text = ft.Text('Processing...', size=14)
    global_loading_row = ft.Row([global_loading_ring, global_loading_text], spacing=10, visible=False)

    def start_loading():
        global_loading_row.visible = True
        page.update()

    def stop_loading():
        global_loading_row.visible = False
        page.update()
    method_result = {'method': None, 'results': []}
    device_config_dropdown = ft.Dropdown(label='Existing device', width=250, options=[])
    config_device_name_field = ft.TextField(label='Device Name', width=250, hint_text='Enter new device name')
    config_rows_field = ft.TextField(label='Rows', width=130, keyboard_type=ft.KeyboardType.NUMBER)
    config_cols_field = ft.TextField(label='Columns', width=130, keyboard_type=ft.KeyboardType.NUMBER)
    config_panel_size_field = ft.TextField(label='Panel Size', width=130, keyboard_type=ft.KeyboardType.NUMBER)
    config_criteria_dropdown = ft.Dropdown(label='Criteria', width=130, options=[ft.DropdownOption('A'), ft.DropdownOption('B')], value='A')
    config_status = ft.Text('', size=14)
    config_save_button = ft.Button(content='Save Device Configuration', icon=ft.Icons.SAVE)
    device_config_container = ft.Container(content=ft.Column([ft.Text('Device Configuration', size=20, weight=ft.FontWeight.BOLD), ft.Text('Add a configuration for a new device or edit an existing device.', size=14), ft.Row([device_config_dropdown, config_device_name_field, config_rows_field, config_cols_field, config_panel_size_field, config_criteria_dropdown, config_save_button], spacing=10, vertical_alignment=ft.CrossAxisAlignment.START, wrap=True), config_status], spacing=8), padding=15)
    result_status = ft.Text('', size=14)
    result_device_checkboxes = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)
    result_item_checkboxes = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO)
    show_multiple_checkbox = ft.Checkbox(label='Show multiple', value=False)
    select_all_button = ft.Button(content='Select All', visible=False)
    clear_all_button = ft.Button(content='Clear All', visible=False)
    show_selected_button = ft.Button(content='Show Selected', visible=False)
    result_info = ft.Text('', size=14)
    result_plot_panel = ft.Container(content=ft.Text('Apply the filter to generate the result.'), expand=True, padding=10)
    return locals()

class DeviceConfigController:
    """Manage device selection and editable device configuration UI."""

    def __init__(self, *, ft, page, device_config, save_device_config, device_config_dropdown, config_device_name_field, config_rows_field, config_cols_field, config_panel_size_field, config_criteria_dropdown, config_status, get_detected_devices, update_input_checkboxes):
        self.ft = ft
        self.page = page
        self.device_config = device_config
        self._save_device_config = save_device_config
        self.device_config_dropdown = device_config_dropdown
        self.config_device_name_field = config_device_name_field
        self.config_rows_field = config_rows_field
        self.config_cols_field = config_cols_field
        self.config_panel_size_field = config_panel_size_field
        self.config_criteria_dropdown = config_criteria_dropdown
        self.config_status = config_status
        self._get_detected_devices = get_detected_devices
        self._update_input_checkboxes = update_input_checkboxes

    def get_selected_input_devices(self, device_checkboxes):
        return [checkbox.label for checkbox in device_checkboxes.controls if checkbox.value]

    def update_device_checkboxes(self, device_checkboxes, devices, selection_callback):
        device_checkboxes.controls.clear()
        for device in devices:
            device_checkboxes.controls.append(self.ft.Checkbox(label=device, value=False, on_change=selection_callback))
        if device_checkboxes.controls:
            device_checkboxes.controls[0].value = True
        self.page.update()

    def update_device_config_dropdown(self, devices=None, preferred_device=None):
        if devices is None:
            devices = sorted(self.device_config.keys())
        devices = sorted(set((str(device) for device in devices)) | set(self.device_config.keys()))
        self.device_config_dropdown.options = [self.ft.DropdownOption(device) for device in devices]
        if not devices:
            self.device_config_dropdown.value = None
            self.config_device_name_field.value = ''
            self.config_rows_field.value = ''
            self.config_cols_field.value = ''
            self.config_panel_size_field.value = ''
            self.config_criteria_dropdown.value = 'A'
            return
        if preferred_device in devices:
            selected_device = preferred_device
        elif self.device_config_dropdown.value in devices:
            selected_device = self.device_config_dropdown.value
        else:
            selected_device = devices[0]
        self.device_config_dropdown.value = selected_device
        self.config_device_name_field.value = selected_device
        self.load_config_into_fields(selected_device)

    def load_config_into_fields(self, device):
        device = str(device or '').strip()
        self.config_device_name_field.value = device
        config = self.device_config.get(device)
        if config is None:
            self.config_rows_field.value = ''
            self.config_cols_field.value = ''
            self.config_panel_size_field.value = ''
            self.config_criteria_dropdown.value = 'A'
            return
        self.config_rows_field.value = str(config.get('rows', ''))
        self.config_cols_field.value = str(config.get('cols', ''))
        self.config_panel_size_field.value = str(config.get('panel_size', ''))
        criteria = str(config.get('criteria', 'A')).strip().upper()
        self.config_criteria_dropdown.value = criteria if criteria in ('A', 'B') else 'A'

    def device_config_changed(self, e):
        self.load_config_into_fields(e.control.value)
        self.config_status.value = ''
        self.page.update()

    def save_current_device_config(self, e):
        device = str(self.config_device_name_field.value or '').strip()
        if not device:
            self.config_status.value = 'Enter a Device Name.'
            self.page.update()
            return
        try:
            rows = int(self.config_rows_field.value)
            cols = int(self.config_cols_field.value)
            panel_size = int(self.config_panel_size_field.value)
            criteria = str(self.config_criteria_dropdown.value or '').strip().upper()
            if rows <= 0 or cols <= 0 or panel_size <= 0:
                raise ValueError
            if criteria not in ('A', 'B'):
                raise ValueError
        except (TypeError, ValueError):
            self.config_status.value = 'Rows, Columns and Panel Size must be positive integers.'
            self.page.update()
            return
        self.device_config[device] = {'rows': rows, 'cols': cols, 'panel_size': panel_size, 'criteria': criteria}
        try:
            self._save_device_config(self.device_config)
            self.update_device_config_dropdown(self._get_detected_devices(), preferred_device=device)
            self.config_status.value = f"Configuration saved for '{device}'."
            self.page.update()
        except Exception as ex:
            self.config_status.value = f'Could not save configuration: {ex}'
            self.page.update()

class BatchHelperController:
    """Manage Batch Helper discovery, selection, and Method 2 assignment."""

    def __init__(self, *, ft, page, batch_source_df=None, batch_overview_df=None, selected_batch_lots=None, batch_checkbox_map=None, method2_reference_batch_keys=None, method2_current_batch_keys=None, get_selected_devices, start_date, end_date, analysis_mode, batch_status, batch_table, batch_helper_container, batch_show_button, batch_use_button, batch_add_a_button, batch_add_b_button, method2_batch_assignment, method2_reference_list, method2_current_list, data_selection_mode):
        self.ft = ft
        self.page = page
        self.batch_source_df = batch_source_df
        self.batch_overview_df = batch_overview_df if batch_overview_df is not None else pd.DataFrame()
        self.selected_batch_lots = selected_batch_lots if selected_batch_lots is not None else []
        self.batch_checkbox_map = batch_checkbox_map if batch_checkbox_map is not None else {}
        self.method2_reference_batch_keys = method2_reference_batch_keys if method2_reference_batch_keys is not None else set()
        self.method2_current_batch_keys = method2_current_batch_keys if method2_current_batch_keys is not None else set()
        self.get_selected_devices = get_selected_devices
        self.start_date = start_date
        self.end_date = end_date
        self.analysis_mode = analysis_mode
        self.batch_status = batch_status
        self.batch_table = batch_table
        self.batch_helper_container = batch_helper_container
        self.batch_show_button = batch_show_button
        self.batch_use_button = batch_use_button
        self.batch_add_a_button = batch_add_a_button
        self.batch_add_b_button = batch_add_b_button
        self.method2_batch_assignment = method2_batch_assignment
        self.method2_reference_list = method2_reference_list
        self.method2_current_list = method2_current_list
        self.data_selection_mode = data_selection_mode

    def selected_batch_keys(self):
        return selected_batch_keys(self.batch_checkbox_map)

    def refresh_batch_helper(self):
        if self.batch_source_df is None or self.batch_source_df.empty:
            self.batch_status.value = 'Upload an Excel file first.'
            self.batch_table.controls.clear()
            self.page.update()
            return
        try:
            selected_devices = self.get_selected_devices()
            if not selected_devices:
                self.batch_overview_df = pd.DataFrame()
                self.batch_status.value = 'Please select at least one device.'
                self.batch_table.controls.clear()
                self.batch_checkbox_map.clear()
                self.page.update()
                return
            self.batch_overview_df = build_batch_overview(self.batch_source_df, start_value=self.start_date.value, end_value=self.end_date.value, devices=selected_devices)
        except Exception as ex:
            self.batch_overview_df = pd.DataFrame()
            self.batch_status.value = f'Batch discovery error: {ex}'
            self.batch_table.controls.clear()
            self.page.update()
            return
        self.batch_table.controls.clear()
        self.batch_checkbox_map.clear()
        valid_batch_keys = set(self.batch_overview_df.get('BatchKey', []))
        self.selected_batch_lots[:] = [item for item in self.selected_batch_lots if item.get('BatchKey') in valid_batch_keys]
        if self.batch_overview_df.empty:
            self.batch_status.value = 'No batches could be identified.'
            self.page.update()
            return
        self.batch_table.controls.append(self.ft.Container(content=self.ft.Row([self.ft.Container(width=35), self.ft.Container(self.ft.Text('USMDevice', weight=self.ft.FontWeight.BOLD), width=190), self.ft.Container(self.ft.Text('Batch', weight=self.ft.FontWeight.BOLD), width=90), self.ft.Container(self.ft.Text('Start', weight=self.ft.FontWeight.BOLD), width=145), self.ft.Container(self.ft.Text('End', weight=self.ft.FontWeight.BOLD), width=145), self.ft.Container(self.ft.Text('Lots', weight=self.ft.FontWeight.BOLD), width=55), self.ft.Container(self.ft.Text('Strips', weight=self.ft.FontWeight.BOLD), width=65), self.ft.Container(self.ft.Text('MESLotID(s)', weight=self.ft.FontWeight.BOLD), expand=True)], spacing=5), padding=5))
        selected_keys = {item.get('BatchKey') for item in self.selected_batch_lots}
        for _, row in self.batch_overview_df.iterrows():
            batch_key = row['BatchKey']
            checkbox = self.ft.Checkbox(value=batch_key in selected_keys)
            self.batch_checkbox_map[batch_key] = checkbox
            row_control = self.ft.Container(content=self.ft.Row([self.ft.Container(checkbox, width=35), self.ft.Container(self.ft.Text(str(row['USMDevice'])), width=190), self.ft.Container(self.ft.Text(str(row['Batch'])), width=90), self.ft.Container(self.ft.Text(str(row['Start'])), width=145), self.ft.Container(self.ft.Text(str(row['End'])), width=145), self.ft.Container(self.ft.Text(f"{int(row['Lots'])}"), width=55), self.ft.Container(self.ft.Text(f"{int(row['Strips'])}"), width=65), self.ft.Container(self.ft.Text(str(row['MESLotID(s)']), size=12), expand=True)], spacing=5, vertical_alignment=self.ft.CrossAxisAlignment.START), padding=5)
            self.batch_table.controls.append(row_control)
        selected_devices = self.get_selected_devices()
        self.batch_status.value = f"{len(self.batch_overview_df):,} batch group(s) found for {', '.join(selected_devices)} | {self.start_date.value} to {self.end_date.value}. Select the batch(es) to use for analysis."
        is_m2 = self.analysis_mode.value == 'Method 2 — Batch Comparison'
        self.method2_batch_assignment.visible = is_m2
        self.batch_add_a_button.visible = is_m2
        self.batch_add_b_button.visible = is_m2
        self.batch_use_button.content = 'Apply A/B Selection' if is_m2 else 'Use Selected Batches'
        self.refresh_method2_batch_assignment()
        self.batch_helper_container.visible = True
        self.batch_show_button.visible = False
        self.page.update()

    def method2_batch_display_name(self, batch_key):
        row = self.batch_overview_df[self.batch_overview_df['BatchKey'].astype(str) == str(batch_key)]
        if row.empty:
            return str(batch_key)
        value = row.iloc[0]
        return f"{value['USMDevice']} | {value['Batch']} | {pd.Timestamp(value['Start']).strftime('%Y-%m-%d')}"

    def refresh_method2_batch_assignment(self):
        self.method2_reference_list.controls.clear()
        self.method2_current_list.controls.clear()
        for key in sorted(self.method2_reference_batch_keys, key=str):
            self.method2_reference_list.controls.append(self.ft.Text(self.method2_batch_display_name(key), size=12))
        for key in sorted(self.method2_current_batch_keys, key=str):
            self.method2_current_list.controls.append(self.ft.Text(self.method2_batch_display_name(key), size=12))
        self.method2_batch_assignment.visible = self.analysis_mode.value == 'Method 2 — Batch Comparison'

    def add_selected_to_method2(self, group):
        selected = set(self.selected_batch_keys())
        if not selected:
            self.batch_status.value = 'Please select one or more batches in the table first.'
            self.page.update()
            return
        if group == 'A':
            self.method2_reference_batch_keys.update(selected)
            self.method2_current_batch_keys.difference_update(selected)
            label = 'Reference A'
        else:
            self.method2_current_batch_keys.update(selected)
            self.method2_reference_batch_keys.difference_update(selected)
            label = 'Current B'
        self.refresh_method2_batch_assignment()
        self.batch_status.value = f'Added {len(selected):,} batch(es) to {label}. A={len(self.method2_reference_batch_keys):,}, B={len(self.method2_current_batch_keys):,}.'
        self.page.update()

    def clear_method2_group(self, group):
        if group == 'A':
            self.method2_reference_batch_keys.clear()
        else:
            self.method2_current_batch_keys.clear()
        self.refresh_method2_batch_assignment()
        self.page.update()

    def apply_method2_batches(self):
        if not self.method2_reference_batch_keys:
            self.batch_status.value = 'Please assign at least one batch to Reference A.'
            self.page.update()
            return
        if not self.method2_current_batch_keys:
            self.batch_status.value = 'Please assign at least one batch to Current B.'
            self.page.update()
            return
        self.batch_status.value = f'Reference A: {len(self.method2_reference_batch_keys):,} batch(es) | Current B: {len(self.method2_current_batch_keys):,} batch(es). Selection applied.'
        self.batch_helper_container.visible = False
        self.batch_show_button.visible = True
        self.page.update()

    def batch_lot_records(self, batch_keys):
        return batch_lot_records(self.batch_overview_df, batch_keys)

    def use_selected_batches(self):
        selected_keys = self.selected_batch_keys()
        if not selected_keys:
            self.batch_status.value = 'Please select at least one batch.'
            self.page.update()
            return
        self.selected_batch_lots[:] = self.batch_lot_records(selected_keys)
        self.batch_status.value = f'Selected {len(selected_keys):,} batch(es) / {len(self.selected_batch_lots):,} lot(s). Batch Helper hidden.'
        self.batch_helper_container.visible = False
        self.batch_show_button.visible = True
        self.page.update()

    def show_batch_helper(self):
        self.batch_helper_container.visible = True
        self.batch_show_button.visible = False
        self.page.update()

    def hide_batch_helper(self):
        self.batch_helper_container.visible = False
        self.batch_show_button.visible = True
        self.page.update()

    def select_all_batches(self):
        for checkbox in self.batch_checkbox_map.values():
            checkbox.value = True
        self.page.update()

    def clear_all_batches(self):
        for checkbox in self.batch_checkbox_map.values():
            checkbox.value = False
        self.page.update()

    def data_selection_mode_changed(self, e):
        if e.control.value == 'Batch Helper':
            if self.batch_source_df is None or self.batch_source_df.empty:
                self.batch_status.value = 'Upload an Excel file first.'
                self.batch_helper_container.visible = False
                self.batch_show_button.visible = False
            else:
                self.refresh_batch_helper()
                self.batch_show_button.visible = False
        else:
            self.batch_helper_container.visible = False
            self.batch_show_button.visible = False
            self.selected_batch_lots.clear()
            self.method2_reference_batch_keys.clear()
            self.method2_current_batch_keys.clear()
            self.refresh_method2_batch_assignment()
            self.batch_status.value = ''
        self.page.update()

    def batch_input_selection_changed(self, e):
        if self.data_selection_mode.value != 'Batch Helper':
            return
        self.method2_reference_batch_keys.clear()
        self.method2_current_batch_keys.clear()
        self.selected_batch_lots.clear()
        self.refresh_batch_helper()

    def batch_use_clicked(self, e):
        if self.analysis_mode.value == 'Method 2 — Batch Comparison':
            self.apply_method2_batches()
        else:
            self.use_selected_batches()

class AnalysisSelectionController:
    """Manage analysis-mode and Method 1 selection UI state."""

    def __init__(self, *, page: Any, analysis_mode: Any, n_day_field: Any, method1_top_strip_dropdown: Any, method_top_bins_field: Any, method_n_iter_field: Any, method_anomaly_checkbox: Any, method_anomaly_z_field: Any, method2_reference_start: Any, method2_reference_end: Any, method2_current_start: Any, method2_current_end: Any, method2_p_threshold: Any, method2_batch_assignment: Any, batch_add_a_button: Any, batch_add_b_button: Any, data_selection_mode: Any, batch_use_button: Any, batch_helper_container: Any, result_selection_panel: Any, result_plot_panel: Any, result_info: Any, method_export_button: Any, get_batch_source_df: Callable[[], Any], get_device_data: Callable[[], dict], get_method_result: Callable[[], dict], refresh_batch_helper: Callable[[], Any], update_result_items: Callable[[], Any]) -> None:
        self.page = page
        self.analysis_mode = analysis_mode
        self.n_day_field = n_day_field
        self.method1_top_strip_dropdown = method1_top_strip_dropdown
        self.method_top_bins_field = method_top_bins_field
        self.method_n_iter_field = method_n_iter_field
        self.method_anomaly_checkbox = method_anomaly_checkbox
        self.method_anomaly_z_field = method_anomaly_z_field
        self.method2_reference_start = method2_reference_start
        self.method2_reference_end = method2_reference_end
        self.method2_current_start = method2_current_start
        self.method2_current_end = method2_current_end
        self.method2_p_threshold = method2_p_threshold
        self.method2_batch_assignment = method2_batch_assignment
        self.batch_add_a_button = batch_add_a_button
        self.batch_add_b_button = batch_add_b_button
        self.data_selection_mode = data_selection_mode
        self.batch_use_button = batch_use_button
        self.batch_helper_container = batch_helper_container
        self.result_selection_panel = result_selection_panel
        self.result_plot_panel = result_plot_panel
        self.result_info = result_info
        self.method_export_button = method_export_button
        self.get_batch_source_df = get_batch_source_df
        self.get_device_data = get_device_data
        self.get_method_result = get_method_result
        self.refresh_batch_helper = refresh_batch_helper
        self.update_result_items = update_result_items

    def analysis_mode_changed(self, e: Any) -> None:
        mode = e.control.value
        is_m1 = mode == 'Method 1 — DBSCAN Clustering'
        is_m2 = mode == 'Method 2 — Batch Comparison'
        is_method = is_m1 or is_m2
        self.n_day_field.visible = mode == 'N-Day Aggregation'
        self.method1_top_strip_dropdown.visible = is_m1
        for control in [self.method_top_bins_field, self.method_n_iter_field]:
            control.visible = is_m2
        for control in [self.method_anomaly_checkbox, self.method_anomaly_z_field]:
            control.visible = is_method
        for control in [self.method2_reference_start, self.method2_reference_end, self.method2_current_start, self.method2_current_end]:
            control.visible = False
        self.method2_p_threshold.visible = is_m2
        use_batch_helper = is_m2 and self.data_selection_mode.value == 'Batch Helper'
        self.method2_batch_assignment.visible = use_batch_helper
        self.batch_add_a_button.visible = use_batch_helper
        self.batch_add_b_button.visible = use_batch_helper
        if is_m2:
            self.data_selection_mode.value = 'Batch Helper'
            self.batch_use_button.content = 'Apply A/B Selection'
            batch_source_df = self.get_batch_source_df()
            if batch_source_df is not None and (not batch_source_df.empty):
                self.refresh_batch_helper()
            else:
                self.batch_helper_container.visible = False
        else:
            self.batch_use_button.content = 'Use Selected Batches'
        self.result_selection_panel.visible = not is_m2
        self.method_export_button.visible = False
        if not is_m2 and self.get_device_data():
            self.update_result_items()
        if is_m2 and self.data_selection_mode.value == 'Batch Helper' and (self.get_batch_source_df() is not None):
            self.refresh_batch_helper()
        self.page.update()

    def method1_top_strip_changed(self, e: Any) -> None:
        if self.analysis_mode.value == 'Method 1 — DBSCAN Clustering' and self.get_method_result().get('results'):
            self.update_result_items()
            self.result_plot_panel.content = self._text('Select a device and StripID to display the DBSCAN cluster plot.')
            self.result_info.value = f'Plot selection: {self.method1_top_strip_dropdown.value}'
            self.page.update()

    def _text(self, message: str) -> Any:
        return ft.Text(message)

class ResultDisplayController:
    """Own result-selection callbacks and result rendering state."""

    def __init__(self, *, ft, asyncio_module, page, analysis_mode, method1_top_strip_dropdown, device_data, method_result, result_device_checkboxes, result_item_checkboxes, result_plot_panel, result_info, item_label, select_all_button, clear_all_button, show_selected_button, show_multiple_checkbox, device_config, plot_device_map, create_method1_plot, plot_control_from_bytes):
        self.ft = ft
        self.asyncio = asyncio_module
        self.page = page
        self.analysis_mode = analysis_mode
        self.method1_top_strip_dropdown = method1_top_strip_dropdown
        self.device_data = device_data
        self.method_result = method_result
        self.result_device_checkboxes = result_device_checkboxes
        self.result_item_checkboxes = result_item_checkboxes
        self.result_plot_panel = result_plot_panel
        self.result_info = result_info
        self.item_label = item_label
        self.select_all_button = select_all_button
        self.clear_all_button = clear_all_button
        self.show_selected_button = show_selected_button
        self.show_multiple_checkbox = show_multiple_checkbox
        self.device_config = device_config
        self.plot_device_map = plot_device_map
        self.create_method1_plot = create_method1_plot
        self.plot_control_from_bytes = plot_control_from_bytes

    def create_plot(self, plot_df, device_name, title_text, subtitle_text=None):
        return create_device_plot(plot_df, device_name, title_text, subtitle_text, self.device_config, self.plot_device_map)

    def _get_method1_result_for_device(self, device):
        for criteria, devices, result in self.method_result.get('results', []):
            if str(device) in {str(item) for item in devices}:
                return (criteria, result)
        return (None, None)

    def update_result_device_checkboxes(self, devices):
        self.result_device_checkboxes.controls.clear()
        for device in devices:
            self.result_device_checkboxes.controls.append(self.ft.Checkbox(label=device, value=False, on_change=self.result_device_changed))
        if self.result_device_checkboxes.controls:
            self.result_device_checkboxes.controls[0].value = True

    def select_all_items(self, e):
        for checkbox in self.result_device_checkboxes.controls:
            checkbox.value = True
        self.update_result_items()
        for checkbox in self.result_item_checkboxes.controls:
            checkbox.value = True
        self.result_info.value = 'All devices and items selected.'
        self.page.update()

    def clear_all_items(self, e):
        for checkbox in self.result_device_checkboxes.controls:
            checkbox.value = False
        for checkbox in self.result_item_checkboxes.controls:
            checkbox.value = False
        self.result_plot_panel.content = self.ft.Text('Select device(s) and item(s), then click Show Selected.')
        self.result_info.value = ''
        self.page.update()

    def display_single_plot(self, device, item):
        if device not in self.device_data:
            return
        selected_data = self.device_data[device]
        mode = self.analysis_mode.value
        if mode == 'Method 1 — DBSCAN Clustering':
            criteria, method1_result = self._get_method1_result_for_device(device)
            if method1_result is None:
                return
            plot_df = selected_data['df']
            png_bytes = self.create_method1_plot(method1_result, device=device, criteria=criteria, strip_id=item, plot_df=plot_df)
            if not png_bytes:
                return
            method1_result['plot_png'] = png_bytes
            method1_result['plot_strip_id'] = item
            method1_result['plot_device'] = str(device)
            return self.plot_control_from_bytes(png_bytes)
        if mode == 'Individual Strip':
            strip_col = selected_data['strip_col']
            plot_df = selected_data['df'][selected_data['df'][strip_col].astype(str) == str(item)]
            if plot_df.empty:
                return
            image_base64 = self.create_plot(plot_df=plot_df, device_name=device, title_text=f'{device} - Strip {item}', subtitle_text=f'Reject records: {len(plot_df):,}')
            return self.plot_control_from_bytes(base64.b64decode(image_base64))
        groups = selected_data.get('groups', {})
        if item not in groups:
            return
        group = groups[item]
        plot_df = group['df']
        image_base64 = self.create_plot(plot_df=plot_df, device_name=device, title_text=f"{device} - {group['label']}", subtitle_text=f"Number of strips: {group['number_of_strips']:,} | Reject records: {len(plot_df):,}")
        return self.plot_control_from_bytes(base64.b64decode(image_base64))

    def get_result_selected_devices(self):
        result_device_checkboxes = self.result_device_checkboxes
        return [checkbox.label for checkbox in result_device_checkboxes.controls if checkbox.value]

    def get_result_selected_items(self):
        result_item_checkboxes = self.result_item_checkboxes
        return [checkbox.label for checkbox in result_item_checkboxes.controls if checkbox.value]

    def update_result_items(self):
        ft = self.ft
        page = self.page
        analysis_mode = self.analysis_mode
        method1_top_strip_dropdown = self.method1_top_strip_dropdown
        device_data = self.device_data
        method_result = self.method_result
        result_device_checkboxes = self.result_device_checkboxes
        result_item_checkboxes = self.result_item_checkboxes
        item_label = self.item_label
        select_all_button = self.select_all_button
        clear_all_button = self.clear_all_button
        show_multiple_checkbox = self.show_multiple_checkbox
        show_selected_button = self.show_selected_button
        result_item_changed = self.result_item_changed
        get_result_selected_devices = self.get_result_selected_devices
        result_item_checkboxes.controls.clear()
        selected_devices = get_result_selected_devices()
        if not selected_devices:
            result_item_checkboxes.visible = False
            select_all_button.visible = False
            clear_all_button.visible = False
            page.update()
            return
        mode = analysis_mode.value
        if mode in ('Individual Strip', 'Method 1 — DBSCAN Clustering'):
            item_label.value = 'StripID'
            if mode == 'Method 1 — DBSCAN Clustering':
                top_selection = str(method1_top_strip_dropdown.value or 'Top 10')
                top_n = None if top_selection == 'All' else int(top_selection.replace('Top ', ''))
                multi_device = len(selected_devices) > 1
                for device in selected_devices:
                    selected_data = device_data.get(device)
                    if selected_data is None:
                        continue
                    strip_col = selected_data['strip_col']
                    all_date_range_strips = selected_data['df'][strip_col].dropna().astype(str).unique().tolist()
                    if top_n is None:
                        strip_ids = sorted(all_date_range_strips, key=str)
                    else:
                        strip_ids = []
                        for _, result_devices, result in method_result.get('results', []):
                            if str(device) not in [str(d) for d in result_devices]:
                                continue
                            strip_summary = result.get('strip_summary', pd.DataFrame())
                            if not strip_summary.empty and 'StripID' in strip_summary.columns:
                                device_summary = strip_summary[strip_summary['USMDevice'].astype(str) == str(device)] if 'USMDevice' in strip_summary.columns else strip_summary.iloc[0:0]
                                if not device_summary.empty:
                                    ranked = device_summary.head(top_n)
                                    strip_ids = ranked['StripID'].dropna().astype(str).tolist()
                            break
                        if not strip_ids:
                            strip_ids = sorted(all_date_range_strips, key=str)[:top_n]
                    for strip_id in strip_ids:
                        label = f'{device} | {strip_id}' if multi_device else str(strip_id)
                        result_item_checkboxes.controls.append(ft.Checkbox(label=label, value=False, on_change=result_item_changed))
            else:
                all_strip_ids = set()
                for device in selected_devices:
                    if device not in device_data:
                        continue
                    selected_data = device_data[device]
                    strip_col = selected_data['strip_col']
                    strip_ids = selected_data['df'][strip_col].dropna().astype(str).unique().tolist()
                    all_strip_ids.update(strip_ids)
                all_strip_ids = sorted(all_strip_ids, key=str)
                for strip_id in all_strip_ids:
                    result_item_checkboxes.controls.append(ft.Checkbox(label=strip_id, value=False, on_change=result_item_changed))
        else:
            item_label.value = 'Period'
            all_periods = set()
            for device in selected_devices:
                selected_data = device_data.get(device)
                if selected_data is None:
                    continue
                groups = selected_data.get('groups', {})
                all_periods.update(groups.keys())
            all_periods = sorted(all_periods, key=str)
            for period in all_periods:
                result_item_checkboxes.controls.append(ft.Checkbox(label=period, value=False, on_change=result_item_changed))
        result_item_checkboxes.visible = True
        select_all_button.visible = show_multiple_checkbox.value
        clear_all_button.visible = show_multiple_checkbox.value
        show_selected_button.visible = show_multiple_checkbox.value
        page.update()

    async def display_single_result(self):
        ft = self.ft
        asyncio = self.asyncio
        page = self.page
        analysis_mode = self.analysis_mode
        result_plot_panel = self.result_plot_panel
        result_info = self.result_info
        device_data = self.device_data
        display_single_plot = self.display_single_plot
        get_result_selected_devices = self.get_result_selected_devices
        get_result_selected_items = self.get_result_selected_items
        result_item_checkboxes = self.result_item_checkboxes
        selected_devices = get_result_selected_devices()
        if not selected_devices:
            result_plot_panel.content = ft.Text('Please select a device.')
            result_info.value = ''
            page.update()
            return
        selected_items = get_result_selected_items()
        if not selected_items:
            if result_item_checkboxes.controls:
                first_item = result_item_checkboxes.controls[0].label
                result_item_checkboxes.controls[0].value = True
                selected_items = [first_item]
            else:
                result_plot_panel.content = ft.Text('No result available.')
                page.update()
                return
        device = selected_devices[0]
        item = selected_items[0]
        result_plot_panel.content = ft.Column([ft.ProgressRing(width=40, height=40), ft.Text('Processing...')], horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER, expand=True)
        result_info.value = f'Generating plot: {device} - {item}'
        page.update()
        try:
            image = await asyncio.to_thread(display_single_plot, device, item)
            if image is None:
                result_plot_panel.content = ft.Text('No data available for this selection.')
            else:
                result_plot_panel.content = image
            if analysis_mode.value in ('Individual Strip', 'Method 1 — DBSCAN Clustering'):
                if device in device_data:
                    strip_col = device_data[device]['strip_col']
                    plot_df = device_data[device]['df']
                    plot_df = plot_df[plot_df[strip_col].astype(str) == str(item)]
                    result_info.value = f'Device: {device}\nStripID: {item}\nReject records: {len(plot_df):,}'
            else:
                group = device_data[device]['groups'][item]
                result_info.value = f"Device: {device}\n{group['label']}\nNumber of strips: {group['number_of_strips']:,}\nReject records: {len(group['df']):,}"
        except Exception as ex:
            result_plot_panel.content = ft.Text(f'Plot error: {ex}')
            result_info.value = ''
        page.update()

    async def display_multiple_results(self):
        ft = self.ft
        asyncio = self.asyncio
        page = self.page
        analysis_mode = self.analysis_mode
        result_plot_panel = self.result_plot_panel
        result_info = self.result_info
        device_data = self.device_data
        display_single_plot = self.display_single_plot
        get_result_selected_devices = self.get_result_selected_devices
        get_result_selected_items = self.get_result_selected_items
        selected_devices = get_result_selected_devices()
        selected_items = get_result_selected_items()
        if not selected_devices:
            result_plot_panel.content = ft.Text('Please select at least one device.')
            result_info.value = ''
            page.update()
            return
        if not selected_items:
            result_plot_panel.content = ft.Text('Please select at least one StripID / Period.')
            result_info.value = ''
            page.update()
            return
        if analysis_mode.value == 'Method 1 — DBSCAN Clustering' and len(selected_devices) > 1:
            total_expected = sum((1 for item in selected_items if ' | ' in str(item) and str(item).split(' | ', 1)[0] in selected_devices))
        else:
            total_expected = len(selected_devices) * len(selected_items)
        result_plot_panel.content = ft.Column([ft.ProgressRing(width=40, height=40), ft.Text('Processing...')], horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER, expand=True)
        result_info.value = f'Generating {total_expected:,} plot(s)...'
        page.update()
        plot_controls = []
        total_plots = 0
        processed = 0
        try:
            if analysis_mode.value == 'Method 1 — DBSCAN Clustering' and len(selected_devices) > 1:
                for item in selected_items:
                    try:
                        if ' | ' not in str(item):
                            continue
                        item_device, strip_id = str(item).split(' | ', 1)
                        if item_device not in selected_devices:
                            continue
                        processed += 1
                        result_info.value = f'Generating plot {processed}/{total_expected}: {item_device} - {strip_id}'
                        page.update()
                        image = await asyncio.to_thread(display_single_plot, item_device, strip_id)
                        if image is None:
                            continue
                        plot_controls.append(ft.Text(f'{item_device} - {strip_id}', size=18, weight=ft.FontWeight.BOLD))
                        plot_controls.append(image)
                        plot_controls.append(ft.Divider())
                        total_plots += 1
                    except Exception as ex:
                        plot_controls.append(ft.Text(f'Plot error - {item}: {ex}'))
            else:
                for device in selected_devices:
                    if device not in device_data:
                        continue
                    for item in selected_items:
                        try:
                            processed += 1
                            result_info.value = f'Generating plot {processed}/{total_expected}: {device} - {item}'
                            page.update()
                            image = await asyncio.to_thread(display_single_plot, device, item)
                            if image is None:
                                continue
                            plot_controls.append(ft.Text(f'{device} - {item}', size=18, weight=ft.FontWeight.BOLD))
                            plot_controls.append(image)
                            plot_controls.append(ft.Divider())
                            total_plots += 1
                        except Exception as ex:
                            plot_controls.append(ft.Text(f'Plot error - {device} / {item}: {ex}'))
            if total_plots == 0:
                result_plot_panel.content = ft.Text('No plots available for the selected device(s) / item(s).')
                result_info.value = ''
            else:
                result_plot_panel.content = ft.Column(plot_controls, spacing=10, scroll=ft.ScrollMode.AUTO)
                result_info.value = f'Showing {total_plots:,} plot(s)'
        except Exception as ex:
            result_plot_panel.content = ft.Text(f'Plot error: {ex}')
            result_info.value = ''
        page.update()

    async def show_selected(self, e):
        show_multiple_checkbox = self.show_multiple_checkbox
        display_multiple_results = self.display_multiple_results
        display_single_result = self.display_single_result
        if show_multiple_checkbox.value:
            await display_multiple_results()
        else:
            await display_single_result()

    async def result_device_changed(self, e):
        show_multiple_checkbox = self.show_multiple_checkbox
        result_device_checkboxes = self.result_device_checkboxes
        ft = self.ft
        update_result_items = self.update_result_items
        display_single_result = self.display_single_result
        result_plot_panel = self.result_plot_panel
        page = self.page
        if not show_multiple_checkbox.value:
            selected = False
            for checkbox in result_device_checkboxes.controls:
                if checkbox is e.control:
                    checkbox.value = True
                    selected = True
                else:
                    checkbox.value = False
            update_result_items()
            await display_single_result()
        else:
            update_result_items()
            result_plot_panel.content = ft.Text('Select device(s) and item(s), then click Show Selected.')
        page.update()

    async def result_item_changed(self, e):
        show_multiple_checkbox = self.show_multiple_checkbox
        result_item_checkboxes = self.result_item_checkboxes
        display_single_result = self.display_single_result
        page = self.page
        if not show_multiple_checkbox.value:
            for checkbox in result_item_checkboxes.controls:
                if checkbox is e.control:
                    checkbox.value = True
                else:
                    checkbox.value = False
            await display_single_result()
        page.update()

    async def show_multiple_changed(self, e):
        show_multiple_checkbox = self.show_multiple_checkbox
        select_all_button = self.select_all_button
        clear_all_button = self.clear_all_button
        show_selected_button = self.show_selected_button
        result_plot_panel = self.result_plot_panel
        result_device_checkboxes = self.result_device_checkboxes
        update_result_items = self.update_result_items
        display_single_result = self.display_single_result
        page = self.page
        ft = self.ft
        multiple = show_multiple_checkbox.value
        if multiple:
            select_all_button.visible = True
            clear_all_button.visible = True
            show_selected_button.visible = True
            result_plot_panel.content = ft.Text('Select device(s) and StripID / period(s), then click Show Selected.')
        else:
            select_all_button.visible = False
            clear_all_button.visible = False
            show_selected_button.visible = False
            first_device = None
            for checkbox in result_device_checkboxes.controls:
                if first_device is None:
                    checkbox.value = True
                    first_device = checkbox.label
                else:
                    checkbox.value = False
            update_result_items()
            await display_single_result()
        page.update()

class ApplicationWorkflow:

    def __init__(self, context):
        """Own the application-level workflow while keeping functional UI controllers separate."""
        for name, value in context.items():
            setattr(self, name, value)

    async def pick_excel(self, e):
        files = await self.file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=['xlsx', 'xls'])
        if not files:
            return
        selected_file = files[0]
        self.selected_file_text.value = f'Selected file: {selected_file.name}'
        try:
            loaded = load_excel_data(selected_file.path, device_config=self.device_config, data_cleaning=data_cleaning, add_effective_unit=add_effective_unit)
            self.df = loaded.dataframe
            self.batch_helper_controller.batch_source_df = loaded.batch_source_df
            self.batch_helper_controller.selected_batch_lots.clear()
            self.batch_helper_controller.method2_reference_batch_keys.clear()
            self.batch_helper_controller.method2_current_batch_keys.clear()
            self.batch_helper_controller.batch_overview_df = pd.DataFrame()
            self.detected_devices.clear()
            self.detected_devices.extend(loaded.all_devices)
            self.upload_status.value = f'File loaded successfully. {len(self.df):,} TA records found. {len(loaded.all_devices)} device(s) detected.'
            self.update_device_checkboxes(loaded.analysis_devices)
            preferred_device = next((device for device in loaded.all_devices if device not in self.device_config), loaded.all_devices[0] if loaded.all_devices else None)
            self.update_device_config_dropdown(loaded.all_devices, preferred_device=preferred_device)
            missing_devices = [device for device in loaded.all_devices if device not in self.device_config]
            if missing_devices:
                self.config_status.value = 'Configuration required for: ' + ', '.join(missing_devices)
            else:
                self.config_status.value = 'All detected devices have a saved configuration.'
            self.device_data.clear()
            self.result_device_checkboxes.controls.clear()
            self.result_item_checkboxes.controls.clear()
            self.result_plot_panel.content = ft.Text('Apply the filter to generate the result.')
            self.result_info.value = ''
            self.result_status.value = ''
        except Exception as ex:
            self.df = None
            self.batch_helper_controller.batch_source_df = None
            self.batch_helper_controller.batch_overview_df = pd.DataFrame()
            self.batch_helper_controller.selected_batch_lots = []
            self.batch_helper_controller.method2_reference_batch_keys.clear()
            self.batch_helper_controller.method2_current_batch_keys.clear()
            self.batch_table.controls.clear()
            self.batch_helper_container.visible = False
            self.batch_show_button.visible = False
            self.upload_status.value = f'Error loading file: {ex}'
        self.page.update()

    def _method_int(self, field, name):
        try:
            value = int(field.value)
            if value <= 0:
                raise ValueError
            return value
        except Exception:
            raise ValueError(f'{name} must be a positive integer.')

    def _method_float(self, field, name):
        try:
            return float(field.value)
        except Exception:
            raise ValueError(f'{name} must be numeric.')

    def _get_analysis_population(self, devices, start_value=None, end_value=None, use_batches=True):
        if start_value is None:
            start_value = self.start_date.value
        if end_value is None:
            end_value = self.end_date.value
        return get_analysis_population_core(self.df, devices, start_value, end_value, use_batches, self.data_selection_mode.value, self.batch_helper_controller.selected_batch_lots)

    def _get_batch_lot_ids(self, batch_keys, device=None):
        """Resolve Batch Helper keys to MESLotID values."""
        keys = {str(key) for key in batch_keys}
        if device is not None:
            prefix = str(device) + ' | '
            keys = {key for key in keys if key.startswith(prefix)}
        return {record['MESLotID'] for record in self.batch_helper_controller.batch_lot_records(keys)}

    def _apply_anomaly_preprocessing(self, dataframe, z_threshold):
        return apply_anomaly_preprocessing(dataframe, self.method_anomaly_checkbox.value, z_threshold, self.device_config, add_strip_anomaly_feature)

    def get_device_criteria(self, device):
        config = self.device_config.get(str(device), {})
        criteria = str(config.get('criteria', '')).strip().upper()
        if criteria not in ('A', 'B'):
            raise ValueError(f"Device '{device}' does not have a valid Criteria A/B configuration.")
        return criteria

    def group_devices_by_criteria(self, selected_devices):
        groups = {}
        for device in selected_devices:
            criteria = self.get_device_criteria(device)
            groups.setdefault(criteria, []).append(device)
        return groups

    def _show_method_results(self):
        show_method_results(ft=ft, result_plot_panel=self.result_plot_panel, method_result=self.method_result, plot_control_from_bytes=_plot_control_from_bytes)

    async def run_method_analysis(self):
        if self.df is None:
            self.method_status.value = 'Please upload an Excel file first.'
            self.page.update()
            return
        selected_devices = self.get_selected_input_devices()
        if not selected_devices:
            self.method_status.value = 'Please select at least one device.'
            self.page.update()
            return
        try:
            mode = self.analysis_mode.value
            z_threshold = self._method_float(self.method_anomaly_z_field, 'Z-score')
            criteria_groups = self.group_devices_by_criteria(selected_devices)
            if mode == 'Method 1 — DBSCAN Clustering':
                eps = self.METHOD1_EPS
                min_samples = self.METHOD1_MIN_SAMPLES
                min_cluster_size = self.METHOD1_MIN_CLUSTER_SIZE
                results = []
                for criteria, devices in sorted(criteria_groups.items()):
                    group_population = self._get_analysis_population(devices, self.start_date.value, self.end_date.value)
                    if group_population.empty:
                        continue
                    group_df = self._apply_anomaly_preprocessing(group_population, z_threshold)
                    if group_df.empty:
                        continue
                    result = await asyncio.to_thread(run_method1, df=group_df, device_config=self.device_config, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size)
                    result['plot_png'] = None
                    results.append((criteria, devices, result))
                if not results:
                    raise ValueError('No Criteria group contains data for Method 1 DBSCAN.')
                self.method_result['method'] = 'Method 1'
                self.method_result['results'] = results
                self.device_data.clear()
                devices_with_data = []
                selected_population = self._get_analysis_population(selected_devices, self.start_date.value, self.end_date.value)
                for device in selected_devices:
                    device_df = selected_population[selected_population['USMDevice'].astype(str) == str(device)].copy()
                    if device_df.empty:
                        continue
                    device_df = self._apply_anomaly_preprocessing(device_df, z_threshold)
                    if device_df.empty:
                        continue
                    devices_with_data.append(device)
                    self.device_data[device] = {'df': device_df, 'strip_col': get_strip_id_column(device_df)}
                self.update_result_device_checkboxes(devices_with_data)
                self.update_result_items()
                self.result_selection_panel.visible = True
                self.method_status.value = 'Method 1 DBSCAN completed: ' + '; '.join((f"Criteria {c} ({', '.join(ds)})" for c, ds, _ in results))
                self.method_export_button.visible = True
                self.result_plot_panel.content = ft.Text('Select a device and StripID to display the DBSCAN cluster plot.')
                if self.data_selection_mode.value == 'Batch Helper':
                    self.result_info.value = f'Batch selection: {len(self.batch_helper_controller.selected_batch_lots):,} lot(s) | Plot selection: {self.method1_top_strip_dropdown.value}'
                else:
                    self.result_info.value = f'Plot selection: {self.method1_top_strip_dropdown.value}'
            elif mode == 'Method 2 — Batch Comparison':
                top_n = self._method_int(self.method_top_bins_field, 'Top bins')
                if self.data_selection_mode.value != 'Batch Helper':
                    raise ValueError('Method 2 requires Batch Helper selection for Reference A and Current B.')
                if not self.batch_helper_controller.method2_reference_batch_keys:
                    raise ValueError('Please assign at least one batch to Reference A.')
                if not self.batch_helper_controller.method2_current_batch_keys:
                    raise ValueError('Please assign at least one batch to Current B.')
                results = []
                for criteria, devices in sorted(criteria_groups.items()):
                    base_population = self._get_analysis_population(devices, self.start_date.value, self.end_date.value, use_batches=False)
                    ref_frames = []
                    cur_frames = []
                    for device in devices:
                        source = base_population[base_population['USMDevice'].astype(str) == str(device)].copy()
                        if source.empty:
                            continue
                        ref_lots = self._get_batch_lot_ids(self.batch_helper_controller.method2_reference_batch_keys, device=device)
                        cur_lots = self._get_batch_lot_ids(self.batch_helper_controller.method2_current_batch_keys, device=device)
                        ref_df = source[source['MESLotID'].astype(str).isin(ref_lots)].copy()
                        cur_df = source[source['MESLotID'].astype(str).isin(cur_lots)].copy()
                        if not ref_df.empty:
                            ref_frames.append(ref_df)
                        if not cur_df.empty:
                            cur_frames.append(cur_df)
                    if not ref_frames or not cur_frames:
                        continue
                    ref_df = pd.concat(ref_frames, ignore_index=True)
                    cur_df = pd.concat(cur_frames, ignore_index=True)
                    ref_df['_method2_group'] = 'Reference A'
                    cur_df['_method2_group'] = 'Current B'
                    combined = pd.concat([ref_df, cur_df], ignore_index=True)
                    if self.method_anomaly_checkbox.value:
                        criteria_map = {str(device): str(config.get('criteria', '')).strip().upper() for device, config in self.device_config.items() if isinstance(config, dict)}
                        combined = add_strip_anomaly_feature(combined, z_threshold=z_threshold, device_config=self.device_config, criteria_map=criteria_map)
                        combined = combined[~combined['strip_level_anomaly']].copy()
                    strips_a = combined.loc[combined['_method2_group'] == 'Reference A', 'StripID'].dropna().unique().tolist()
                    strips_b = combined.loc[combined['_method2_group'] == 'Current B', 'StripID'].dropna().unique().tolist()
                    combined = combined.drop(columns=['_method2_group'])
                    if not strips_a or not strips_b:
                        continue
                    result = await asyncio.to_thread(run_method2, df=combined, devices=devices, device_config=self.device_config, strips_a=strips_a, strips_b=strips_b, n_perm=self._method_int(self.method_n_iter_field, 'Iterations'), top_n=top_n, p_threshold=self._method_float(self.method2_p_threshold, 'p threshold'))
                    result['plot_png'] = await asyncio.to_thread(create_method2_plot, result, ', '.join(devices), criteria=criteria)
                    results.append((criteria, devices, result))
                if not results:
                    raise ValueError('No Criteria group contains both Reference and Current data.')
                self.method_result['method'] = 'Method 2'
                self.method_result['results'] = results
                status_parts = []
                for criteria, devices, result in results:
                    stats = result['stats']
                    status_parts.append(f"Criteria {criteria} | JSD={stats['actual_jsd']:.6f} | p%={stats['p_pct']:.2f} | {('SIGNIFICANT' if stats['significant'] else 'NOT SIGNIFICANT')}")
                self.method_status.value = 'Method 2 completed: ' + '; '.join(status_parts)
                self.method_export_button.visible = True
                self._show_method_results()
            else:
                return
        except Exception as ex:
            self.method_result['method'] = None
            self.method_result['results'] = []
            self.method_export_button.visible = False
            self.method_status.value = f'Method error: {ex}'
        self.page.update()

    async def export_method_excel(self, e):
        if not self.method_result['results']:
            self.method_status.value = 'Run a method first.'
            self.page.update()
            return
        try:
            save = await self.method_file_picker.save_file(dialog_title='Export Method Excel', file_name='Method1_DBSCAN_results.xlsx' if self.method_result['method'] == 'Method 1' else 'Method2_results.xlsx', allowed_extensions=['xlsx'])
            if not save:
                return
            if isinstance(save, str):
                path = save
            else:
                path = getattr(save, 'path', None)
                if not path:
                    raise ValueError('No export file path was returned.')
            if self.method_result['method'] == 'Method 1':
                export_method1_excel(self.method_result['results'], path)
            else:
                export_method2_excel(self.method_result['results'], path)
            self.method_status.value = f'Excel exported: {path}'
        except Exception as ex:
            self.method_status.value = f'Export error: {ex}'
        self.page.update()

    def _set_filter_message(self, message, plot_message=None):
        self.result_status.value = message
        if plot_message is not None:
            self.result_plot_panel.content = ft.Text(plot_message)
        self.page.update()

    def _validate_filter_inputs(self):
        if self.df is None:
            self._set_filter_message('Please upload an Excel file first.')
            return None
        selected_devices = self.get_selected_input_devices()
        if not selected_devices:
            self._set_filter_message('Please select at least one device.')
            return None
        missing_config = [device for device in selected_devices if device not in self.device_config]
        if missing_config:
            self.result_status.value = 'Please save device configuration for: ' + ', '.join(missing_config)
            self.config_status.value = 'Configuration required for: ' + ', '.join(missing_config)
            self.update_device_config_dropdown(self.detected_devices, preferred_device=missing_config[0])
            self.page.update()
            return None
        try:
            start = pd.to_datetime(self.start_date.value).normalize()
            end = pd.to_datetime(self.end_date.value).normalize()
        except Exception:
            self._set_filter_message('Invalid date format. Please use YYYY-MM-DD.')
            return None
        if start > end:
            self._set_filter_message('Start date cannot be later than end date.')
            return None
        n_days = None
        if self.analysis_mode.value == 'N-Day Aggregation':
            try:
                n_days = int(self.n_day_field.value)
                if n_days <= 0:
                    raise ValueError
            except Exception:
                self._set_filter_message('Number of days must be a positive integer.')
                return None
        return (selected_devices, start, end, n_days)

    def _prepare_filter_population(self, selected_devices, start, end):
        filtered = self._get_analysis_population(selected_devices, start, end)
        if filtered.empty:
            self._set_filter_message('No data found for the selected device(s) and date range.', 'No data available.')
            return None
        if self.method_anomaly_checkbox.value:
            filtered = self._apply_anomaly_preprocessing(filtered, self._method_float(self.method_anomaly_z_field, 'Z-score'))
            if filtered.empty:
                self._set_filter_message('No anomalous strips found in the selected population.', 'No anomalous strips available.')
                return None
        return filtered

    async def _finalize_filter_results(self, devices_with_data):
        if not devices_with_data:
            self._set_filter_message('No data found for the selected devices.')
            return
        self.update_result_device_checkboxes(devices_with_data)
        self.update_result_items()
        total_records = sum((len(self.device_data[device]['df']) for device in devices_with_data))
        self.result_status.value = f'Devices: {len(devices_with_data):,} | TA records: {total_records:,}'
        if self.show_multiple_checkbox.value:
            self.result_plot_panel.content = ft.Text('Select device(s) and item(s), then click Show Selected.')
        else:
            await self.display_single_result()
        self.page.update()

    async def _run_standard_filter(self):
        inputs = self._validate_filter_inputs()
        if inputs is None:
            return
        selected_devices, start, end, n_days = inputs
        filtered = self._prepare_filter_population(selected_devices, start, end)
        if filtered is None:
            return
        strip_col, devices_with_data = prepare_device_analysis_data(filtered, selected_devices, self.analysis_mode.value, start, end, n_days, self.device_data, get_strip_id_column)
        if strip_col is None:
            self._set_filter_message('StripID column was not found in the Excel file.')
            return
        await self._finalize_filter_results(devices_with_data)

    async def run_filter(self, e):
        self.start_loading()
        await asyncio.sleep(0)
        try:
            if self.analysis_mode.value in ('Method 1 — DBSCAN Clustering', 'Method 2 — Batch Comparison'):
                await self.run_method_analysis()
                return
            self.result_status.value = ''
            self.result_plot_panel.content = ft.Text('Processing...')
            self.result_info.value = ''
            self.page.update()
            await self._run_standard_filter()
        finally:
            self.stop_loading()

def load_device_config():
    """Load saved device configuration, falling back to defaults."""
    if not DEVICE_CONFIG_FILE.exists():
        return DEFAULT_DEVICE_CONFIG.copy()
    try:
        with open(DEVICE_CONFIG_FILE, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        if not isinstance(saved_config, dict):
            return DEFAULT_DEVICE_CONFIG.copy()
        config = {device: values.copy() for device, values in DEFAULT_DEVICE_CONFIG.items()}
        for device, values in saved_config.items():
            if isinstance(values, dict):
                base = config.get(device, {'criteria': 'A'}).copy()
                base.update(values)
                config[device] = base
        return config
    except Exception:
        return DEFAULT_DEVICE_CONFIG.copy()

def save_device_config(config):
    """Save device configuration to JSON."""
    with open(DEVICE_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def get_strip_id_column(dataframe):
    """Return the StripID column using common casing variants."""
    possible_names = ['StripID', 'StripId', 'stripID', 'stripId', 'STRIPID']
    for name in possible_names:
        if name in dataframe.columns:
            return name
    for column in dataframe.columns:
        if str(column).lower() == 'stripid':
            return column
    return None

def build_application(page: ft.Page):
    configure_page(page)
    device_config = load_device_config()
    df = None
    device_data = {}
    detected_devices = []
    controls = create_app_controls(page)
    title = controls['title']
    upload_status = controls['upload_status']
    selected_file_text = controls['selected_file_text']
    device_checkboxes = controls['device_checkboxes']
    start_date = controls['start_date']
    end_date = controls['end_date']
    data_selection_mode = controls['data_selection_mode']
    batch_status = controls['batch_status']
    method2_reference_list = controls['method2_reference_list']
    method2_current_list = controls['method2_current_list']
    method2_clear_a_button = controls['method2_clear_a_button']
    method2_clear_b_button = controls['method2_clear_b_button']
    method2_reference_column = controls['method2_reference_column']
    method2_current_column = controls['method2_current_column']
    method2_batch_assignment = controls['method2_batch_assignment']
    batch_table = controls['batch_table']
    batch_find_button = controls['batch_find_button']
    batch_select_all_button = controls['batch_select_all_button']
    batch_clear_all_button = controls['batch_clear_all_button']
    batch_use_button = controls['batch_use_button']
    batch_hide_button = controls['batch_hide_button']
    batch_helper_header = controls['batch_helper_header']
    batch_helper_container = controls['batch_helper_container']
    batch_show_button = controls['batch_show_button']
    analysis_mode = controls['analysis_mode']
    n_day_field = controls['n_day_field']
    METHOD1_EPS = controls['METHOD1_EPS']
    METHOD1_MIN_SAMPLES = controls['METHOD1_MIN_SAMPLES']
    METHOD1_MIN_CLUSTER_SIZE = controls['METHOD1_MIN_CLUSTER_SIZE']
    method1_top_strip_dropdown = controls['method1_top_strip_dropdown']
    method_top_bins_field = controls['method_top_bins_field']
    method_n_iter_field = controls['method_n_iter_field']
    method_anomaly_checkbox = controls['method_anomaly_checkbox']
    method_anomaly_z_field = controls['method_anomaly_z_field']
    method2_reference_start = controls['method2_reference_start']
    method2_reference_end = controls['method2_reference_end']
    method2_current_start = controls['method2_current_start']
    method2_current_end = controls['method2_current_end']
    method2_p_threshold = controls['method2_p_threshold']
    method_export_button = controls['method_export_button']
    method_status = controls['method_status']
    global_loading_ring = controls['global_loading_ring']
    global_loading_text = controls['global_loading_text']
    global_loading_row = controls['global_loading_row']
    start_loading = controls['start_loading']
    stop_loading = controls['stop_loading']
    method_result = controls['method_result']
    device_config_dropdown = controls['device_config_dropdown']
    config_device_name_field = controls['config_device_name_field']
    config_rows_field = controls['config_rows_field']
    config_cols_field = controls['config_cols_field']
    config_panel_size_field = controls['config_panel_size_field']
    config_criteria_dropdown = controls['config_criteria_dropdown']
    config_status = controls['config_status']
    config_save_button = controls['config_save_button']
    device_config_container = controls['device_config_container']
    result_status = controls['result_status']
    result_device_checkboxes = controls['result_device_checkboxes']
    result_item_checkboxes = controls['result_item_checkboxes']
    show_multiple_checkbox = controls['show_multiple_checkbox']
    select_all_button = controls['select_all_button']
    clear_all_button = controls['clear_all_button']
    show_selected_button = controls['show_selected_button']
    result_info = controls['result_info']
    result_plot_panel = controls['result_plot_panel']
    batch_add_a_button = ft.Button(content='Add Selected → A', visible=False)
    batch_add_b_button = ft.Button(content='Add Selected → B', visible=False)
    batch_helper_header.controls.insert(4, batch_add_a_button)
    batch_helper_header.controls.insert(5, batch_add_b_button)
    device_config_controller = DeviceConfigController(ft=ft, page=page, device_config=device_config, save_device_config=save_device_config, device_config_dropdown=device_config_dropdown, config_device_name_field=config_device_name_field, config_rows_field=config_rows_field, config_cols_field=config_cols_field, config_panel_size_field=config_panel_size_field, config_criteria_dropdown=config_criteria_dropdown, config_status=config_status, get_detected_devices=lambda: detected_devices, update_input_checkboxes=lambda devices: None)
    batch_helper_controller = BatchHelperController(ft=ft, page=page, get_selected_devices=lambda: device_config_controller.get_selected_input_devices(device_checkboxes), start_date=start_date, end_date=end_date, analysis_mode=analysis_mode, batch_status=batch_status, batch_table=batch_table, batch_helper_container=batch_helper_container, batch_show_button=batch_show_button, batch_use_button=batch_use_button, batch_add_a_button=batch_add_a_button, batch_add_b_button=batch_add_b_button, method2_batch_assignment=method2_batch_assignment, method2_reference_list=method2_reference_list, method2_current_list=method2_current_list, data_selection_mode=data_selection_mode)
    get_selected_input_devices = lambda: device_config_controller.get_selected_input_devices(device_checkboxes)
    update_device_checkboxes = lambda devices: device_config_controller.update_device_checkboxes(device_checkboxes, devices, batch_helper_controller.batch_input_selection_changed)
    update_device_config_dropdown = device_config_controller.update_device_config_dropdown
    refresh_batch_helper = batch_helper_controller.refresh_batch_helper
    device_config_dropdown.on_select = device_config_controller.device_config_changed
    config_save_button.on_click = device_config_controller.save_current_device_config
    data_selection_mode.on_select = batch_helper_controller.data_selection_mode_changed
    start_date.on_change = batch_helper_controller.batch_input_selection_changed
    end_date.on_change = batch_helper_controller.batch_input_selection_changed
    batch_find_button.on_click = lambda e: batch_helper_controller.refresh_batch_helper()
    batch_select_all_button.on_click = lambda e: batch_helper_controller.select_all_batches()
    batch_clear_all_button.on_click = lambda e: batch_helper_controller.clear_all_batches()
    batch_use_button.on_click = batch_helper_controller.batch_use_clicked
    batch_hide_button.on_click = lambda e: batch_helper_controller.hide_batch_helper()
    batch_show_button.on_click = lambda e: batch_helper_controller.show_batch_helper()
    batch_add_a_button.on_click = lambda e: batch_helper_controller.add_selected_to_method2('A')
    batch_add_b_button.on_click = lambda e: batch_helper_controller.add_selected_to_method2('B')
    method2_clear_a_button.on_click = lambda e: batch_helper_controller.clear_method2_group('A')
    method2_clear_b_button.on_click = lambda e: batch_helper_controller.clear_method2_group('B')
    file_picker = ft.FilePicker()
    page.services.append(file_picker)
    method_file_picker = ft.FilePicker()
    page.services.append(method_file_picker)
    input_device_container = ft.Container(content=ft.Column([ft.Text('Device', size=16, weight=ft.FontWeight.BOLD), device_checkboxes], spacing=5), width=380)
    item_label = ft.Text('StripID', size=16, weight=ft.FontWeight.BOLD)
    result_selection_panel = ft.Container(content=ft.Column([ft.Text('Selection', size=20, weight=ft.FontWeight.BOLD), ft.Text('Devices', size=16, weight=ft.FontWeight.BOLD), result_device_checkboxes, show_multiple_checkbox, item_label, result_item_checkboxes, ft.Row([select_all_button, clear_all_button], spacing=5), show_selected_button, result_info], spacing=10), width=360, padding=15)
    result_row = ft.Row([result_selection_panel, ft.VerticalDivider(width=1), result_plot_panel], vertical_alignment=ft.CrossAxisAlignment.START, expand=False)
    result_display_controller = ResultDisplayController(ft=ft, asyncio_module=asyncio, page=page, analysis_mode=analysis_mode, method1_top_strip_dropdown=method1_top_strip_dropdown, device_data=device_data, method_result=method_result, result_device_checkboxes=result_device_checkboxes, result_item_checkboxes=result_item_checkboxes, result_plot_panel=result_plot_panel, result_info=result_info, item_label=item_label, select_all_button=select_all_button, clear_all_button=clear_all_button, show_selected_button=show_selected_button, show_multiple_checkbox=show_multiple_checkbox, device_config=device_config, plot_device_map=plot_device_map, create_method1_plot=create_method1_plot, plot_control_from_bytes=lambda image_bytes, width=1100, height=None: plot_control_from_bytes(ft, image_bytes, width, height))
    get_result_selected_devices = result_display_controller.get_result_selected_devices
    get_result_selected_items = result_display_controller.get_result_selected_items
    update_result_items = result_display_controller.update_result_items
    update_result_device_checkboxes = result_display_controller.update_result_device_checkboxes
    select_all_items = result_display_controller.select_all_items
    clear_all_items = result_display_controller.clear_all_items
    display_single_result = result_display_controller.display_single_result
    display_multiple_results = result_display_controller.display_multiple_results
    show_selected = result_display_controller.show_selected
    result_device_changed = result_display_controller.result_device_changed
    result_item_changed = result_display_controller.result_item_changed
    show_multiple_changed = result_display_controller.show_multiple_changed
    workflow_context = {'METHOD1_EPS': METHOD1_EPS, 'METHOD1_MIN_CLUSTER_SIZE': METHOD1_MIN_CLUSTER_SIZE, 'METHOD1_MIN_SAMPLES': METHOD1_MIN_SAMPLES, 'analysis_mode': analysis_mode, 'batch_helper_container': batch_helper_container, 'batch_helper_controller': batch_helper_controller, 'batch_show_button': batch_show_button, 'batch_table': batch_table, 'config_status': config_status, 'data_selection_mode': data_selection_mode, 'detected_devices': detected_devices, 'device_config': device_config, 'device_data': device_data, 'df': df, 'display_single_result': display_single_result, 'end_date': end_date, 'file_picker': file_picker, 'get_selected_input_devices': get_selected_input_devices, 'method1_top_strip_dropdown': method1_top_strip_dropdown, 'method2_p_threshold': method2_p_threshold, 'method_anomaly_checkbox': method_anomaly_checkbox, 'method_anomaly_z_field': method_anomaly_z_field, 'method_export_button': method_export_button, 'method_file_picker': method_file_picker, 'method_n_iter_field': method_n_iter_field, 'method_result': method_result, 'method_status': method_status, 'method_top_bins_field': method_top_bins_field, 'n_day_field': n_day_field, 'page': page, 'result_device_checkboxes': result_device_checkboxes, 'result_info': result_info, 'result_item_checkboxes': result_item_checkboxes, 'result_plot_panel': result_plot_panel, 'result_selection_panel': result_selection_panel, 'result_status': result_status, 'selected_file_text': selected_file_text, 'show_multiple_checkbox': show_multiple_checkbox, 'start_date': start_date, 'start_loading': start_loading, 'stop_loading': stop_loading, 'update_device_checkboxes': update_device_checkboxes, 'update_device_config_dropdown': update_device_config_dropdown, 'update_result_device_checkboxes': update_result_device_checkboxes, 'update_result_items': update_result_items, 'upload_status': upload_status}
    workflow = ApplicationWorkflow(workflow_context)
    method_export_button.on_click = workflow.export_method_excel
    analysis_selection_controller = AnalysisSelectionController(page=page, analysis_mode=analysis_mode, n_day_field=n_day_field, method1_top_strip_dropdown=method1_top_strip_dropdown, method_top_bins_field=method_top_bins_field, method_n_iter_field=method_n_iter_field, method_anomaly_checkbox=method_anomaly_checkbox, method_anomaly_z_field=method_anomaly_z_field, method2_reference_start=method2_reference_start, method2_reference_end=method2_reference_end, method2_current_start=method2_current_start, method2_current_end=method2_current_end, method2_p_threshold=method2_p_threshold, method2_batch_assignment=method2_batch_assignment, batch_add_a_button=batch_add_a_button, batch_add_b_button=batch_add_b_button, data_selection_mode=data_selection_mode, batch_use_button=batch_use_button, batch_helper_container=batch_helper_container, result_selection_panel=result_selection_panel, result_plot_panel=result_plot_panel, result_info=result_info, method_export_button=method_export_button, get_batch_source_df=lambda: batch_helper_controller.batch_source_df, get_device_data=lambda: device_data, get_method_result=lambda: method_result, refresh_batch_helper=refresh_batch_helper, update_result_items=update_result_items)
    analysis_mode.on_select = analysis_selection_controller.analysis_mode_changed
    method1_top_strip_dropdown.on_select = analysis_selection_controller.method1_top_strip_changed
    show_selected_button.on_click = show_selected
    show_multiple_checkbox.on_change = show_multiple_changed
    select_all_button.on_click = select_all_items
    clear_all_button.on_click = clear_all_items
    update_device_config_dropdown()
    upload_button = ft.Button(content='Upload Excel', icon=ft.Icons.UPLOAD_FILE, on_click=workflow.pick_excel)
    plot_button = ft.Button(content='Apply Filter & Plot', icon=ft.Icons.FILTER_ALT, on_click=workflow.run_filter)
    page.add(title, ft.Divider(), ft.Text('1. Data Input', size=24, weight=ft.FontWeight.BOLD), ft.Row([upload_button, selected_file_text], spacing=15), upload_status, ft.Divider(), ft.Text('2. Analysis Selection', size=24, weight=ft.FontWeight.BOLD), ft.Row([input_device_container, start_date, end_date, data_selection_mode], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START), ft.Row([analysis_mode, n_day_field, batch_show_button], spacing=12), batch_helper_container, ft.Row([method1_top_strip_dropdown, method_top_bins_field, method_n_iter_field], spacing=10, wrap=True), ft.Row([method2_reference_start, method2_reference_end, method2_current_start, method2_current_end, method2_p_threshold], spacing=10, wrap=True), ft.Row([method_anomaly_checkbox, method_anomaly_z_field, method_export_button], spacing=10, wrap=True), global_loading_row, method_status, device_config_container, ft.Container(content=plot_button, padding=ft.Padding.only(top=10, bottom=10)), result_status, ft.Divider(), ft.Text('3. Result', size=24, weight=ft.FontWeight.BOLD), result_row)
