import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict, deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import math

       st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")
       st.title("📊 Gestión de Proyectos - Seguimiento y Control")
       
       archivo_excel = st.file_uploader("Subir archivo Excel con hojas Tareas, Recursos y Dependencias", type=["xlsx"])
       tab1, tab2, tab3, tab4 = st.tabs(["Inicio", "Diagrama Gantt", "Recursos", "Presupuesto"])
       
       # --- Refactored Data Loading and Session State Management ---
       if archivo_excel:
           # Load data only when a new file is uploaded
           if 'archivo_hash' not in st.session_state or st.session_state.archivo_hash != hash(archivo_excel.getvalue()):
               st.session_state.archivo_hash = hash(archivo_excel.getvalue())
               try:
                   st.session_state.tareas_df_original = pd.read_excel(archivo_excel, sheet_name='Tareas')
                   st.session_state.recursos_df = pd.read_excel(archivo_excel, sheet_name='Recursos')
                   st.session_state.dependencias_df = pd.read_excel(archivo_excel, sheet_name='Dependencias')
       
                   # Initial data processing and setup for calculation
                   st.session_state.tareas_df = st.session_state.tareas_df_original.copy()
       
                   # Normalize columns and create initial duration
                   if 'TARIFA' in st.session_state.recursos_df.columns:
                       st.session_state.recursos_df['TARIFA'] = pd.to_numeric(st.session_state.recursos_df['TARIFA'], errors='coerce').fillna(0)
       
                   # Ensure date columns are datetime objects
                   for col in ['FECHAINICIO','FECHAFIN']:
                        st.session_state.tareas_df[col] = pd.to_datetime(st.session_state.tareas_df[col], dayfirst=True, errors='coerce')
       
                   st.session_state.tareas_df['PREDECESORAS'] = st.session_state.tareas_df['PREDECESORAS'].fillna('').astype(str)
                   st.session_state.tareas_df['DURACION'] = (st.session_state.tareas_df['FECHAFIN'] - st.session_state.tareas_df['FECHAINICIO']).dt.days.fillna(0).astype(int)
                   st.session_state.tareas_df.loc[st.session_state.tareas_df['DURACION'] < 0, 'DURACION'] = 0 # prevent negative duration
       
                   # Initial calculation of dates and critical path
                   st.session_state.tareas_df = calcular_fechas(st.session_state.tareas_df, st) # Pass st for warnings
                   st.session_state.tareas_df = calcular_ruta_critica(st.session_state.tareas_df, st) # Pass st for warnings
       
                   # Store a copy to detect changes later
                   st.session_state.tareas_df_last_calculated = st.session_state.tareas_df.copy()
       
       
               except Exception as e:
                   st.error(f"Error al leer el archivo Excel. Asegúrese de que contiene las hojas 'Tareas', 'Recursos' y 'Dependencias' y que el formato es correcto: {e}")
                   st.stop()
       
       # --- Function Definitions (Kept as they are, but updated to accept st) ---
       
           # Function to calculate dates based on dependencies
           # This function now only calculates based on the provided dependencies
           def calcular_fechas(df, st_session=None):
               df = df.copy()
               df.columns = df.columns.str.strip()
       
               # Ensure date columns are datetime objects before calculation
               for col in ['FECHAINICIO','FECHAFIN']:
                   if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
       
               df['DURACION'] = (df['FECHAFIN'] - df['FECHAINICIO']).dt.days.fillna(0).astype(int)
               df.loc[df['DURACION'] < 0, 'DURACION'] = 0
       
               inicio_rubro = df.set_index('IDRUBRO')['FECHAINICIO'].to_dict()
               fin_rubro = df.set_index('IDRUBRO')['FECHAFIN'].to_dict()
               duracion_rubro = df.set_index('IDRUBRO')['DURACION'].to_dict()
       
               dependencias = defaultdict(list)
               pre_count = defaultdict(int)
               all_task_ids = set(df['IDRUBRO'].tolist())
       
               for _, row in df.iterrows():
                   tarea_id = row['IDRUBRO']
                   predecesoras_str = str(row['PREDECESORAS']).strip()
                   if predecesoras_str not in ['nan', '']:
                       pre_list = predecesoras_str.split(',')
                       for pre_entry in pre_list:
                           pre_entry = pre_entry.strip()
                           match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                           if match:
                               pre_id = int(match.group(1))
                               if pre_id in all_task_ids:
                                   dependencias[pre_id].append(tarea_id)
                                   pre_count[tarea_id] += 1
                               elif st_session: st_session.warning(f"⚠️ Predecesor ID {pre_id} en '{pre_entry}' para tarea {tarea_id} no encontrado.")
                           elif pre_entry != '' and st_session: st_session.warning(f"⚠️ Formato de predecesora '{pre_entry}' no reconocido para tarea {tarea_id}.")
       
       
               queue = deque([tid for tid in all_task_ids if pre_count[tid] == 0])
               inicio_calc = inicio_rubro.copy()
               fin_calc = fin_rubro.copy()
       
               # Forward Pass (Calculating Early Start and Early Finish)
               while queue:
                   tarea_id = queue.popleft()
                   current_row = df[df['IDRUBRO'] == tarea_id].iloc[0]
                   duracion = duracion_rubro.get(tarea_id, 0)
                   predecesoras_str = str(current_row['PREDECESORAS']).strip()
       
                   earliest_start = inicio_calc.get(tarea_id, pd.NaT) # Use original start date as a baseline
       
                   if predecesoras_str not in ['nan', '']:
                        pre_list = predecesoras_str.split(',')
                        for pre_entry in pre_list:
                            pre_entry = pre_entry.strip()
                            match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                            if match:
                                pre_id = int(match.group(1))
                                tipo = match.group(2).upper() if match.group(2) else 'FC'
                                desfase = int(match.group(3)) if match.group(3) else 0
       
                                if pre_id in inicio_calc and pre_id in fin_calc:
                                    inicio_pre = inicio_calc[pre_id]
                                    fin_pre = fin_calc[pre_id]
       
                                    if pd.notna(inicio_pre) and pd.notna(fin_pre):
                                        calculated_start = pd.NaT
                                        if tipo == 'CC':
                                            calculated_start = inicio_pre + timedelta(days=desfase)
                                        elif tipo == 'FC':
                                            calculated_start = fin_pre + timedelta(days=desfase)
                                        elif tipo == 'CF':
                                            # This relationship type (Start of Predecessor to Finish of Successor)
                                            # is tricky for forward pass. We calculate a potential start based on
                                            # when the successor *could* finish based on the predecessor's start.
                                            # The actual start will be determined by the latest of all predecessor dependencies.
                                             pass # CF and FF are primarily used in backward pass for LS/LF
                                        elif tipo == 'FF':
                                            # Same as CF, primarily for backward pass.
                                             pass
                                        else:
                                             if st_session: st_session.warning(f"⚠️ Tipo de relación '{tipo}' no reconocido en '{pre_entry}' para tarea {tarea_id}. Ignorando para el cálculo del ES.")
                                             continue # Skip unrecognized types for ES calculation
       
                                        if pd.notna(calculated_start):
                                            if pd.isna(earliest_start) or calculated_start > earliest_start:
                                                earliest_start = calculated_start
       
                   if pd.notna(earliest_start):
                        inicio_calc[tarea_id] = earliest_start
                        fin_calc[tarea_id] = earliest_start + timedelta(days=duracion)
                   else:
                        # If no valid predecessors or original date, leave as NaT or handle as error
                        inicio_calc[tarea_id] = pd.NaT
                        fin_calc[tarea_id] = pd.NaT
                        if st_session: st_session.warning(f"⚠️ No se pudo determinar la FECHA_INICIO para la tarea {tarea_id}. Verifique las predecesoras y fechas.")
       
       
                   for hijo in dependencias.get(tarea_id, []):
                          pre_count[hijo] -= 1
                          if pre_count[hijo] == 0:
                                 queue.append(hijo)
       
               # Add calculated early dates to the DataFrame
               df['FECHA_INICIO_TEMPRANA'] = df['IDRUBRO'].map(inicio_calc)
               df['FECHA_FIN_TEMPRANA'] = df['IDRUBRO'].map(fin_calc)
       
               # Backward Pass (Calculating Late Start and Late Finish)
               # Need to rebuild graph for backward pass if necessary or use successor map
               successor_map = defaultdict(list)
               predecesoras_map = defaultdict(list) # Need this for backward pass logic
               for _, row in df.iterrows():
                    tarea_id = row['IDRUBRO']
                    predecesoras_str = str(row['PREDECESORAS']).strip()
                    if predecesoras_str not in ['nan', '']:
                        for pre_entry in predecesoras_str.split(','):
                             pre_entry = pre_entry.strip()
                             match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                             if match:
                                 pre_id = int(match.group(1))
                                 tipo_relacion = match.group(2).upper() if match.group(2) else 'FC'
                                 desfase = int(match.group(3)) if match.group(3) else 0
                                 if pre_id in all_task_ids:
                                     successor_map[pre_id].append((tarea_id, tipo_relacion, desfase))
                                     predecesoras_map[tarea_id].append((pre_id, tipo_relacion, desfase))
       
       
               end_tasks_ids = [tid for tid in all_task_ids if tid not in successor_map or not successor_map[tid]]
               project_finish_date = df['FECHA_FIN_TEMPRANA'].max() # Project finish is the max Early Finish
       
               ls, lf = {}, {}
               queue = deque(end_tasks_ids)
       
               # Initialize Late Finish for end tasks
               for tid in end_tasks_ids:
                   lf[tid] = project_finish_date
                   ls[tid] = lf[tid] - timedelta(days=duracion_rubro.get(tid, 0))
       
       
               visited = set() # To prevent infinite loops in case of cycles (though graph should be DAG)
               while queue:
                   v = queue.popleft()
                   if v in visited:
                       continue
                   visited.add(v)
       
                   for u, tipo_uv, desfase_uv in predecesoras_map.get(v, []):
                       duration_u = duracion_rubro.get(u, 0)
       
                       calculated_lf_u = pd.NaT
       
                       # Calculate potential Late Finish for predecessor u based on successor v
                       if v in ls and v in lf:
                           if tipo_uv == 'CC':
                               # LS(v) = ES(u) + lag => LS(v) - lag = ES(u) => LF(u) = LS(v) - lag + duration(u)
                               if pd.notna(ls.get(v)):
                                    calculated_lf_u = ls[v] - timedelta(days=desfase_uv) + timedelta(days=duration_u)
                           elif tipo_uv == 'FC':
                               # LS(v) = EF(u) + lag => LS(v) - lag = EF(u) => LF(u) = LS(v) - lag
                               if pd.notna(ls.get(v)):
                                    calculated_lf_u = ls[v] - timedelta(days=desfase_uv)
                           elif tipo_uv == 'CF':
                                # LF(v) = ES(u) + lag => LF(v) - lag = ES(u) => LF(u) = LF(v) - lag + duration(u)
                                if pd.notna(lf.get(v)):
                                     calculated_lf_u = lf[v] - timedelta(days=desfase_uv) + timedelta(days=duration_u)
                           elif tipo_uv == 'FF':
                                # LF(v) = EF(u) + lag => LF(v) - lag = EF(u) => LF(u) = LF(v) - lag
                                if pd.notna(lf.get(v)):
                                     calculated_lf_u = lf[v] - timedelta(days=desfase_uv)
                           else:
                               if st_session: st_session.warning(f"⚠️ Tipo relación '{tipo_uv}' no reconocido para LF de tarea {u}. Usando FC por defecto.")
                               if pd.notna(ls.get(v)):
                                    calculated_lf_u = ls[v] - timedelta(days=desfase_uv)
       
       
                       if pd.notna(calculated_lf_u):
                           if u not in lf or pd.isna(lf[u]) or calculated_lf_u < lf[u]:
                               lf[u] = calculated_lf_u
                               ls[u] = lf[u] - timedelta(days=duration_u)
       
                       if u in all_task_ids and u not in visited: # Only add valid predecessors not yet visited
                           queue.append(u)
       
               # Add calculated late dates to the DataFrame
               df['FECHA_INICIO_TARDE'] = df['IDRUBRO'].map(ls)
               df['FECHA_FIN_TARDE'] = df['IDRUBRO'].map(lf)
       
               # Ensure original FECHAINICIO and FECHAFIN match calculated early dates initially
               # This might need adjustment if the user wants to *set* the start/end date manually
               # For now, let's keep the original dates unless they are NaT, then use early dates.
               df['FECHAINICIO'] = df.apply(lambda row: row['FECHA_INICIO_TEMPRANA'] if pd.isna(row['FECHAINICIO']) else row['FECHAINICIO'], axis=1)
               df['FECHAFIN'] = df.apply(lambda row: row['FECHA_FIN_TEMPRANA'] if pd.isna(row['FECHAFIN']) else row['FECHAFIN'], axis=1)
       
       
               return df
       
           # Function to calculate critical path and floats
           # This function now relies on the dates calculated by calcular_fechas
           # Manual RUTA_CRITICA editing in the UI will *not* directly change dependencies here.
           # Instead, the UI edit will trigger recalculation based on defined dependencies.
           def calcular_ruta_critica(df, st_session=None):
               df = df.copy()
       
               # Ensure required date columns exist and are datetime
               required_cols = ['FECHA_INICIO_TEMPRANA', 'FECHA_FIN_TEMPRANA', 'FECHA_INICIO_TARDE', 'FECHA_FIN_TARDE']
               for col in required_cols:
                    if col not in df.columns:
                         df[col] = pd.NaT # Add if missing
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                         df[col] = pd.to_datetime(df[col], errors='coerce')
       
       
               # Calculate Total Float (TF = LF - EF or LS - ES)
               df['HOLGURA_TOTAL_TD'] = df['FECHA_FIN_TARDE'] - df['FECHA_FIN_TEMPRANA']
               df['HOLGURA_TOTAL'] = df['HOLGURA_TOTAL_TD'].apply(lambda x: x.days if pd.notna(x) else pd.NA)
       
               # Calculate Free Float (FF = ES_successor - EF_current)
               # This requires iterating through successors for each task
               successor_map = defaultdict(list)
               all_task_ids = set(df['IDRUBRO'].tolist())
               es_dict = df.set_index('IDRUBRO')['FECHA_INICIO_TEMPRANA'].to_dict()
               ef_dict = df.set_index('IDRUBRO')['FECHA_FIN_TEMPRANA'].to_dict()
               duracion_dict = df.set_index('IDRUBRO')['DURACION'].to_dict()
       
       
               for _, row in df.iterrows():
                    tarea_id = row['IDRUBRO']
                    predecesoras_str = str(row['PREDECESORAS']).strip()
                    if predecesoras_str not in ['nan', '']:
                        for pre_entry in predecesoras_str.split(','):
                             pre_entry = pre_entry.strip()
                             match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                             if match:
                                 pre_id = int(match.group(1))
                                 tipo_relacion = match.group(2).upper() if match.group(2) else 'FC'
                                 desfase = int(match.group(3)) if match.group(3) else 0
                                 if pre_id in all_task_ids:
                                     successor_map[pre_id].append((tarea_id, tipo_relacion, desfase))
       
       
               ff_dict = {}
               for tid in all_task_ids:
                   ef_current = ef_dict.get(tid, pd.NaT)
                   if pd.isna(ef_current):
                       ff_dict[tid] = pd.NA
                       continue
       
                   min_succ_start = None
                   for suc_id, tipo, desfase in successor_map.get(tid, []):
                       if suc_id in es_dict and pd.notna(es_dict.get(suc_id)):
                           succ_es = es_dict.get(suc_id)
                           # Calculate the earliest possible start of the successor based on the current task's early finish
                           # This is simplified; a full FF calculation considers the relationship type
                           # For simplicity here, we'll use the successor's ES as the target, adjusted by lag for FF definition (ES_successor - EF_current - lag)
                           # A more precise FF definition depends on the relationship type.
                           # For FC: FF = ES(successor) - EF(current) - lag
                           # For CC: FF = ES(successor) - ES(current) - lag
                           # For CF: FF = LF(successor) - ES(current) - lag - Duration(successor)
                           # For FF: FF = LF(successor) - EF(current) - lag - Duration(successor)
                           # Let's stick to the most common FF definition (FC relationship implied): ES(successor) - EF(current)
                           # Or the definition used in the original code: min_succ_start - ef[tid]
                           # Let's re-implement the original code's FF logic which seems to calculate based on earliest *actual* start of successor, considering lag
                           succ_es_based_on_pred = pd.NaT
                           if tipo == 'CC':
                               # Based on current task's ES (need ES dict here)
                               es_current = es_dict.get(tid, pd.NaT)
                               if pd.notna(es_current):
                                    succ_es_based_on_pred = es_current + timedelta(days=desfase)
                           elif tipo == 'FC':
                               # Based on current task's EF
                               succ_es_based_on_pred = ef_current + timedelta(days=desfase)
                           elif tipo == 'CF':
                                # Based on current task's ES, determines successor's LF, then work backward to successor's ES
                                es_current = es_dict.get(tid, pd.NaT)
                                duration_succ = duracion_dict.get(suc_id, 0)
                                if pd.notna(es_current):
                                     succ_lf_based_on_pred = es_current + timedelta(days=desfase)
                                     succ_es_based_on_pred = succ_lf_based_on_pred - timedelta(days=duration_succ)
                           elif tipo == 'FF':
                                # Based on current task's EF, determines successor's LF, then work backward to successor's ES
                                duration_succ = duracion_dict.get(suc_id, 0)
                                succ_lf_based_on_pred = ef_current + timedelta(days=desfase)
                                succ_es_based_on_pred = succ_lf_based_on_pred - timedelta(days=duration_succ)
                           else:
                               # Default to FC logic if type is unknown
                                if st_session: st_session.warning(f"⚠️ Tipo relación '{tipo}' no reconocido para FF de tarea {tid} con sucesora {suc_id}. Usando FC por defecto.")
                                succ_es_based_on_pred = ef_current + timedelta(days=desfase)
       
       
                           if pd.notna(succ_es_based_on_pred):
                                if min_succ_start is None or succ_es_based_on_pred < min_succ_start:
                                    min_succ_start = succ_es_based_on_pred
       
       
                   if min_succ_start is not None and pd.notna(ef_current):
                       ff_dict[tid] = min_succ_start - ef_current
                   else:
                       ff_dict[tid] = timedelta(days=0) # Or pd.NA? Let's use 0 timedelta if no successors or dates missing
       
               df['HOLGURA_LIBRE_TD'] = df['IDRUBRO'].map(ff_dict)
               df['HOLGURA_LIBRE'] = df['HOLGURA_LIBRE_TD'].apply(lambda x: x.days if pd.notna(x) else pd.NA)
       
               # Determine Critical Path (Total Float close to zero)
               tolerance_days = 1e-9 # Allow for small floating point inaccuracies
               df['RUTA_CRITICA'] = df['HOLGURA_TOTAL'].apply(lambda x: abs(x) < tolerance_days if pd.notna(x) else False)
       
               return df
       
           # --- Removed actualizar_dependencias_por_critica ---
           # The critical path is now an output of the calculation, not a manual input that changes dependencies.
           # The user can still see the critical path in the UI, but changing the boolean won't alter the dependency string.
           # If the user wants to make a task critical, they should adjust its dependencies or dates directly.
       
       # --- UI Tabs ---
           with tab1:
               st.markdown("#### Datos Importados:")
               st.subheader("📋 Tabla Recursos")
               # Use session state dataframe for display
               gb = GridOptionsBuilder.from_dataframe(st.session_state.recursos_df)
               gb.configure_default_column(editable=True)
               grid_options = gb.build()
               custom_css = {
                      ".ag-header": {  # clase del header completo
                      "background-color": "#0D3B66",  # azul oscuro
                      "color": "white",               # texto blanco
                      "font-weight": "bold",
                      "text-align": "center"
                      }
               }
               # Capture changes back to session state
               recursos_grid_response = AgGrid(st.session_state.recursos_df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED,custom_css=custom_css)
               st.session_state.recursos_df = pd.DataFrame(recursos_grid_response['data'])
       
       
               st.subheader("📋 Tabla Dependencias")
               # Use session state dataframe for display
               gb = GridOptionsBuilder.from_dataframe(st.session_state.dependencias_df)
               gb.configure_default_column(editable=True)
               gb.configure_column(
                      "CANTIDAD",
                      type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                      precision=2,  # número de decimales
                      editable=True
               )
               grid_options = gb.build()
               custom_css = {
                      ".ag-header": {  # clase del header completo
                      "background-color": "#0D3B66",  # azul oscuro
                      "color": "white",               # texto blanco
                      "font-weight": "bold",
                      "text-align": "center"
                      }
               }
               # Capture changes back to session state
               dependencias_grid_response = AgGrid(st.session_state.dependencias_df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED,custom_css=custom_css)
               st.session_state.dependencias_df = pd.DataFrame(dependencias_grid_response['data'])
       
               # Removed the redundant Tareas AgGrid from tab1
       
       
           with tab2:
               st.subheader("📋 Tabla Tareas (Cronograma Calculado)")
       
               # Display the calculated tasks_df from session state
               # Allow editing of relevant columns like PREDECESORAS, FECHAINICIO, FECHAFIN
               df_display = st.session_state.tareas_df[['IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                           'FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA',
                           'FECHA_INICIO_TARDE','FECHA_FIN_TARDE','DURACION','HOLGURA_TOTAL','HOLGURA_LIBRE', 'RUTA_CRITICA']].copy()
       
               # Format dates for display in AgGrid
               for col in ['FECHAINICIO','FECHAFIN','FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA','FECHA_INICIO_TARDE','FECHA_FIN_TARDE']:
                    if col in df_display.columns:
                         df_display[col] = df_display[col].dt.strftime('%d/%m/%Y').replace(pd.NaT, '') # Format dates and handle NaT
       
               gb = GridOptionsBuilder.from_dataframe(df_display)
               gb.configure_default_column(editable=False, resizable=True) # Default to not editable
               gb.configure_column("PREDECESORAS", editable=True)
               gb.configure_column("FECHAINICIO", editable=True) # Allow editing original dates
               gb.configure_column("FECHAFIN", editable=True) # Allow editing original dates
               # RUTA_CRITICA is an output, do not allow editing here
               gb.configure_column("RUTA_CRITICA", editable=False)
       
               grid_options = gb.build()
       
               custom_css = {
                   ".ag-header": {
                       "background-color": "#0D3B66",  # azul oscuro
                       "color": "white",               # texto blanco
                       "font-weight": "bold",
                       "text-align": "center"
                   },
                   ".ag-row-no-focus .ag-cell": { # Style for non-critical path rows
                        "background-color": "#E9EEF2", # Light greyish blue
                   },
                   ".ag-row-no-focus .ag-cell.critical-path": { # Style for critical path rows
                       "background-color": "#FFDDC1", # Light orange/peach color
                       "font-weight": "bold",
                   },
                   ".ag-cell.critical-path": { # Fallback style for critical path cells
                        "background-color": "#FFDDC1",
                        "font-weight": "bold",
                   }
               }
       
               # Add rowClassRules for critical path styling
               grid_options['getRowClass'] = "data.RUTA_CRITICA === true ? 'ag-row-no-focus critical-path' : 'ag-row-no-focus'"
       
       
               grid_response = AgGrid(
                   df_display, # Display the formatted dataframe
                   gridOptions=grid_options,
                   update_mode=GridUpdateMode.MODEL_CHANGED,
                   custom_css=custom_css,
                   fit_columns_on_grid_load=True,
                   height=400,
                   key='tareas_grid_tab2' # Add a unique key
               )
       
               df_editado = pd.DataFrame(grid_response['data'])
       
               # Convert dates back to datetime objects from the edited data
               for col in ['FECHAINICIO','FECHAFIN']:
                    if col in df_editado.columns:
                         df_editado[col] = pd.to_datetime(df_editado[col], format='%d/%m/%Y', errors='coerce')
       
               # Update the session state dataframe with edited values
               # Merge edited data back into the main session state tareas_df
               # Only update the columns that were editable in the grid
               cols_to_update = ['IDRUBRO', 'PREDECESORAS', 'FECHAINICIO', 'FECHAFIN']
               # Ensure edited columns exist in the original session state df
               cols_to_update = [col for col in cols_to_update if col in st.session_state.tareas_df.columns and col in df_editado.columns]
       
               if not df_editado.equals(df_display): # Check if any data was actually changed in the grid
                   # Use set_index and update for efficient merging
                   st.session_state.tareas_df = st.session_state.tareas_df.set_index('IDRUBRO').update(
                       df_editado[cols_to_update].set_index('IDRUBRO')
                   ).reset_index()
       
                   # Recalculate dates and critical path after edits
                   st.session_state.tareas_df = calcular_fechas(st.session_state.tareas_df, st)
                   st.session_state.tareas_df = calcular_ruta_critica(st.session_state.tareas_df, st)
       
                   # Update the copy used for change detection
                   st.session_state.tareas_df_last_calculated = st.session_state.tareas_df.copy()
       
               # --- Add Gantt Chart Placeholder ---
               st.subheader("📈 Diagrama de Gantt")
               # Gantt chart generation code will go here in a later step
               st.write("Gantt chart placeholder")
       
       
       # --- Add other tabs placeholders ---
           with tab3:
               st.subheader("👥 Gestión de Recursos")
               st.write("Resource management features will be added here.")
       
           with tab4:
               st.subheader("💰 Presupuesto y Valor Ganado")
               st.write("Budget and earned value features will be added here.")
       
       elif 'tareas_df' in st.session_state:
           # If no file is uploaded but data is in session state, display the tabs
            with tab1:
               st.markdown("#### Datos Importados:")
               st.subheader("📋 Tabla Recursos")
               # Use session state dataframe for display
               gb = GridOptionsBuilder.from_dataframe(st.session_state.recursos_df)
               gb.configure_default_column(editable=True)
               grid_options = gb.build()
               custom_css = {
                      ".ag-header": {  # clase del header completo
                      "background-color": "#0D3B66",  # azul oscuro
                      "color": "white",               # texto blanco
                      "font-weight": "bold",
                      "text-align": "center"
                      }
               }
               # Capture changes back to session state
               recursos_grid_response = AgGrid(st.session_state.recursos_df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED,custom_css=custom_css)
               st.session_state.recursos_df = pd.DataFrame(recursos_grid_response['data'])
       
       
               st.subheader("📋 Tabla Dependencias")
               # Use session state dataframe for display
               gb = GridOptionsBuilder.from_dataframe(st.session_state.dependencias_df)
               gb.configure_default_column(editable=True)
               gb.configure_column(
                      "CANTIDAD",
                      type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                      precision=2,  # número de decimales
                      editable=True
               )
               grid_options = gb.build()
               custom_css = {
                      ".ag-header": {  # clase del header completo
                      "background-color": "#0D3B66",  # azul oscuro
                      "color": "white",               # texto blanco
                      "font-weight": "bold",
                      "text-align": "center"
                      }
               }
               # Capture changes back to session state
               dependencias_grid_response = AgGrid(st.session_state.dependencias_df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED,custom_css=custom_css)
               st.session_state.dependencias_df = pd.DataFrame(dependencias_grid_response['data'])
       
            with tab2:
              st.subheader("📋 Tabla Tareas (Cronograma Calculado)")
       
               # Display the calculated tasks_df from session state
              df_display = st.session_state.tareas_df[['IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                           'FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA',
                           'FECHA_INICIO_TARDE','FECHA_FIN_TARDE','DURACION','HOLGURA_TOTAL','HOLGURA_LIBRE', 'RUTA_CRITICA']].copy()
       
               # Format dates for display in AgGrid
              for col in ['FECHAINICIO','FECHAFIN','FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA','FECHA_INICIO_TARDE','FECHA_FIN_TARDE']:
                    if col in df_display.columns:
                         df_display[col] = df_display[col].dt.strftime('%d/%m/%Y').replace(pd.NaT, '') # Format dates and handle NaT
       
       
              gb = GridOptionsBuilder.from_dataframe(df_display)
              gb.configure_default_column(editable=False, resizable=True) # Default to not editable
              gb.configure_column("PREDECESORAS", editable=True)
              gb.configure_column("FECHAINICIO", editable=True) # Allow editing original dates
              gb.configure_column("FECHAFIN", editable=True) # Allow editing original dates
              # RUTA_CRITICA is an output, do not allow editing here
              gb.configure_column("RUTA_CRITICA", editable=False)
       
       
              grid_options = gb.build()
       
              custom_css = {
                   ".ag-header": {
                       "background-color": "#0D3B66",  # azul oscuro
                       "color": "white",               # texto blanco
                       "font-weight": "bold",
                       "text-align": "center"
                   },
                   ".ag-row-no-focus .ag-cell": { # Style for non-critical path rows
                        "background-color": "#E9EEF2", # Light greyish blue
                   },
                   ".ag-row-no-focus .ag-cell.critical-path": { # Style for critical path rows
                       "background-color": "#FFDDC1", # Light orange/peach color
                       "font-weight": "bold",
                   },
                    ".ag-cell.critical-path": { # Fallback style for critical path cells
                        "background-color": "#FFDDC1",
                        "font-weight": "bold",
                   }
              }
       
              # Add rowClassRules for critical path styling
              grid_options['getRowClass'] = "data.RUTA_CRITICA === true ? 'ag-row-no-focus critical-path' : 'ag-row-no-focus'"
       
              grid_response = AgGrid(
                   df_display, # Display the formatted dataframe
                   gridOptions=grid_options,
                   update_mode=GridUpdateMode.MODEL_CHANGED,
                   custom_css=custom_css,
                   fit_columns_on_grid_load=True,
                   height=400,
                   key='tareas_grid_tab2' # Add a unique key
              )
       
              df_editado = pd.DataFrame(grid_response['data'])
       
               # Convert dates back to datetime objects from the edited data
              for col in ['FECHAINICIO','FECHAFIN']:
                    if col in df_editado.columns:
                         df_editado[col] = pd.to_datetime(df_editado[col], format='%d/%m/%Y', errors='coerce')
       
       
               # Check if the edited dataframe is different from the last calculated one
               # This prevents unnecessary recalculations if the grid state hasn't changed data-wise
               # We need to compare the relevant columns that could trigger a recalculation
              cols_to_compare = ['IDRUBRO', 'PREDECESORAS', 'FECHAINICIO', 'FECHAFIN']
               # Ensure comparison columns exist in both dataframes
              cols_to_compare = [col for col in cols_to_compare if col in st.session_state.tareas_df.columns and col in df_editado.columns and col in st.session_state.tareas_df_last_calculated.columns]
       
              if not df_editado[cols_to_compare].equals(st.session_state.tareas_df_last_calculated[cols_to_compare]):
                    # Update the session state dataframe with edited values
                    # Use set_index and update for efficient merging
                    st.session_state.tareas_df = st.session_state.tareas_df.set_index('IDRUBRO').update(
                        df_editado[cols_to_update].set_index('IDRUBRO')
                    ).reset_index()
       
                    # Recalculate dates and critical path after edits
                    st.session_state.tareas_df = calcular_fechas(st.session_state.tareas_df, st)
                    st.session_state.tareas_df = calcular_ruta_critica(st.session_state.tareas_df, st)
       
                    # Update the copy used for change detection
                    st.session_state.tareas_df_last_calculated = st.session_state.tareas_df.copy()
                    # Rerun to update the displayed grid with new calculations
                    st.rerun()


              dependencias_df = dependencias_df.merge(recursos_df, left_on='RECURSO', right_on='RECURSO', how='left')
              dependencias_df['COSTO'] = dependencias_df['CANTIDAD'] * dependencias_df['TARIFA']
              costos_por_can = dependencias_df.groupby('RUBRO', as_index=False)['COSTO'].sum()
              costos_por_can.rename(columns={'RUBRO': 'RUBRO', 'COSTO': 'COSTO_TOTAL'}, inplace=True)
              tareas_df['RUBRO'] = tareas_df['RUBRO'].str.strip()
              costos_por_can['RUBRO'] = costos_por_can['RUBRO'].str.strip()
              tareas_df = tareas_df.merge(costos_por_can[['RUBRO', 'COSTO_TOTAL']], on='RUBRO', how='left')
              
              import plotly.graph_objects as go
              import plotly.express as px
              from collections import defaultdict
              
              st.subheader("📊 Diagrama de Gantt - Ruta Crítica")
              
              # Determinar columna de costo
              cost_column_name = None
              for col in ['COSTO_TOTAL_RUBRO', 'COSTO_TOTAL_x', 'COSTO_TOTAL']:
                     if col in tareas_df.columns:
                            cost_column_name = col
                            break
              
              if cost_column_name:
                     tareas_df[cost_column_name] = pd.to_numeric(tareas_df[cost_column_name], errors='coerce').fillna(0)
              else:
                     st.warning("⚠️ No se encontró una columna de costos reconocida en el DataFrame. Se creará columna de costos en 0.")
                     tareas_df['COSTO_TOTAL_NUMERICO'] = 0
                     cost_column_name = 'COSTO_TOTAL_NUMERICO'
              
              if 'IDRUBRO' in tareas_df.columns:
                     tareas_df = tareas_df.sort_values(['IDRUBRO'])
              else:
                     st.warning("⚠️ Columna 'IDRUBRO' no encontrada para ordenar.")
              
              tareas_df['y_num'] = range(len(tareas_df))
              tareas_df['y_num_plot'] = tareas_df['y_num'] + 0.01

              dependencias = defaultdict(list)
              predecesoras_map_details = defaultdict(list)
              warnings_list = []
              
              for _, row in tareas_df.iterrows():
                  tarea_id = row['IDRUBRO']
                  predecesoras_str = str(row.get('PREDECESORAS', '')).strip()
                  if predecesoras_str not in ['nan', '']:
                      pre_list = predecesoras_str.split(',')
                      for pre_entry in pre_list:
                          pre_entry = pre_entry.strip()
                          match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                          if match:
                              pre_id = int(match.group(1))
                              tipo_relacion = match.group(2).upper() if match.group(2) else 'FC'
                              desfase = int(match.group(3)) if match.group(3) else 0
                              if pre_id in tareas_df['IDRUBRO'].values:
                                  dependencias[pre_id].append(tarea_id)
                                  predecesoras_map_details[tarea_id].append((pre_id, tipo_relacion, desfase))
                              else:
                                  warnings_list.append(f"Predecesor ID {pre_id} para tarea {tarea_id} no encontrado.")
                          elif pre_entry != '':
                              warnings_list.append(f"Formato de predecesora '{pre_entry}' no reconocido para tarea {tarea_id}.")
              
              if warnings_list:
                  st.warning("⚠️ Advertencias detectadas:\n" + "\n".join(warnings_list))

              inicio_rubro_calc = tareas_df.set_index('IDRUBRO')['FECHAINICIO'].to_dict()
              fin_rubro_calc = tareas_df.set_index('IDRUBRO')['FECHAFIN'].to_dict()
              is_critical_dict = tareas_df.set_index('IDRUBRO')['RUTA_CRITICA'].to_dict()

              fig = go.Figure()
              shapes = []

              color_no_critica_barra = 'lightblue'
              color_critica_barra = 'rgb(255, 133, 133)'
              
              # Agregar barras
              for i, row in tareas_df.iterrows():
                  line_color = color_critica_barra if row.get('RUTA_CRITICA', False) else color_no_critica_barra
                  line_width = 12
                  start_date = row['FECHAINICIO']
                  end_date = row['FECHAFIN']
                  if pd.isna(start_date) or pd.isna(end_date):
                      continue
              
                  valor_costo = float(row.get(cost_column_name, 0))
                  costo_formateado = f"S/ {valor_costo:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                  hover_text = (
                      f"📌 <b>Rubro:</b> {row['RUBRO']}<br>"
                      f"🗓️ <b>Capítulo:</b> {row.get('CAPÍTULO', '')}<br>"
                      f"📅 <b>Inicio:</b> {start_date.strftime('%d/%m/%Y')}<br>"
                      f"🏁 <b>Fin:</b> {end_date.strftime('%d/%m/%Y')}<br>"
                      f"⏱️ <b>Duración:</b> {(end_date - start_date).days} días<br>"
                      f"⏳ <b>Holgura Total:</b> {row.get('HOLGURA_TOTAL', 'N/A')} días<br>"
                      f"💰 <b>Costo:</b> {costo_formateado}"
                  )
                  import numpy as np
                  x_vals = pd.date_range(start=start_date, end=end_date, freq='D')  # un punto por día
                  y_vals = np.full(len(x_vals), row['y_num_plot'])
                  fig.add_trace(go.Scatter(
                         x=x_vals,
                         y=y_vals,
                         mode='lines',
                         line=dict(color=line_color, width=line_width),
                         showlegend=False,
                         hoverinfo='text',
                         text=[hover_text]*len(x_vals),
                  ))
              
              # Función para dibujar flechas de dependencias
              def dibujar_flecha(pre_id, suc_id, tipo_relacion, offset=5):
                     y_pre = tareas_df.loc[tareas_df['IDRUBRO']==pre_id, 'y_num_plot'].values[0]
                     y_suc = tareas_df.loc[tareas_df['IDRUBRO']==suc_id, 'y_num_plot'].values[0]
                     pre_is_critical = is_critical_dict.get(pre_id, False)
                     suc_is_critical = is_critical_dict.get(suc_id, False)
                     arrow_color = 'red' if pre_is_critical and suc_is_critical else 'blue'
                     
                     x_pre_inicio = inicio_rubro_calc.get(pre_id)
                     x_pre_fin = fin_rubro_calc.get(pre_id)
                     x_suc_inicio = inicio_rubro_calc.get(suc_id)
                     x_suc_fin = fin_rubro_calc.get(suc_id)
                     
                     origin_x = x_pre_fin if tipo_relacion in ['FC', 'FF'] else x_pre_inicio
                     connection_x = x_suc_inicio if tipo_relacion in ['FC', 'CC'] else x_suc_fin
                     points_x, points_y = [origin_x], [y_pre]
                     
                     if tipo_relacion in ['CC', 'FC']:
                             elbow1_x, elbow1_y = origin_x - timedelta(days=offset), y_pre
                             elbow2_x, elbow2_y = elbow1_x, y_suc
                             points_x.extend([elbow1_x, elbow2_x, connection_x])
                             points_y.extend([elbow1_y, elbow2_y, y_suc])
                     elif tipo_relacion in ['CF', 'FF']:
                             elbow1_x, elbow1_y = origin_x, y_suc
                             points_x.extend([elbow1_x, connection_x])
                             points_y.extend([elbow1_y, y_suc])

                     arrow_symbol = 'triangle-right'  # valor por defecto
                     if tipo_relacion == 'CC':
                         arrow_symbol = 'triangle-right'
                     elif tipo_relacion == 'CF':
                         arrow_symbol = 'triangle-left'
                     elif tipo_relacion == 'FF':
                         arrow_symbol = 'triangle-left'

                     # Marcador en el predecesor (círculo)
                     fig.add_trace(go.Scattergl(
                         x=[origin_x],
                         y=[y_pre],
                         mode='markers',
                         marker=dict(symbol='circle', size=8, color=arrow_color),
                         hoverinfo='none',
                         showlegend=False,
                     ))
                     
                     # Línea de la flecha (igual que antes)
                     fig.add_trace(go.Scatter(
                         x=points_x,
                         y=points_y,
                         mode='lines',
                         line=dict(color=arrow_color, width=1, dash='dash'),
                         hoverinfo='none',
                         showlegend=False,
                     ))
                     
                     # Marcador en el sucesor (triángulo)
                     fig.add_trace(go.Scattergl(
                         x=[connection_x],
                         y=[y_suc],
                         mode='markers',
                         marker=dict(symbol=arrow_symbol, size=10, color=arrow_color),
                         hoverinfo='none',
                         showlegend=False,
                     ))
                                   
              # Dibujar todas las flechas
              for pre_id, sucesores in dependencias.items():
                  for suc_id in sucesores:
                      tipo_rel = 'FC'
                      for pre_tmp, type_tmp, _ in predecesoras_map_details.get(suc_id, []):
                          if pre_tmp == pre_id:
                              tipo_rel = type_tmp.upper() if type_tmp else 'FC'
                              break
                      dibujar_flecha(pre_id, suc_id, tipo_rel)
              
              # Preparar Y-ticks
              y_ticktext_styled = []
              for y_pos in range(len(tareas_df)):
                  row = tareas_df[tareas_df['y_num'] == y_pos]
                  if not row.empty:
                      rubro_text = row.iloc[0]['RUBRO']
                      y_ticktext_styled.append(rubro_text)
                  else:
                      y_ticktext_styled.append("")

              fecha_min = tareas_df['FECHAINICIO'].min()
              fecha_max = tareas_df['FECHAFIN'].max()
              years = list(range(fecha_min.year, fecha_max.year + 1))
              colors = ['rgba(200,200,200,0.2)', 'rgba(100,100,100,0.2)']  # gris claro y blanco huevo
              shapes_years = []
              
              for i, year in enumerate(years):
                  shapes_years.append(
                      dict(
                          type='rect',
                          xref='x',
                          yref='paper',  # cubre todo el eje Y
                          x0=pd.Timestamp(f'{year}-01-01'),
                          x1=pd.Timestamp(f'{year}-12-31'),
                          y0=0,
                          y1=1,
                          fillcolor=colors[i % 2],
                          opacity=0.5,
                          layer='below',
                          line_width=0,
                      )
                  )

              shapes = shapes + shapes_years
              
              # Layout
              fig.update_layout(
                  xaxis=dict(
                      title='Fechas',
                      side='bottom',
                      dtick='M1',
                      tickangle=-90,
                      showgrid=True,
                      gridcolor='rgba(128,128,128,0.3)',
                      gridwidth=0.5
                  ),
                     
                  yaxis_title='Rubro',
                  yaxis=dict(
                      autorange='reversed',
                      tickvals=tareas_df['y_num_plot'],
                      ticktext=y_ticktext_styled,
                      tickfont=dict(size=10),
                      showgrid=False
                  ),
                  shapes=shapes,
                  height=max(600, len(tareas_df)*25),
                  showlegend=False,
                  plot_bgcolor='white',
                  hovermode='closest'
              )

              fig.update_xaxes(
                  side="top",
                  overlaying="x",
                  dtick='M1',
                  tickangle=-90,
                  showgrid=True,
                  gridcolor='rgba(128,128,128,0.3)',
                  gridwidth=0.5
              )
              
              st.plotly_chart(fig, use_container_width=True)
    
       tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'])
       tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'])

       tareas_df['RUBRO'] = tareas_df['RUBRO'].str.strip()
       dependencias_df['RUBRO'] = dependencias_df['RUBRO'].str.strip()
        
       recursos_tareas_df = dependencias_df.merge(
              tareas_df[['IDRUBRO', 'RUBRO', 'FECHAINICIO', 'FECHAFIN', 'DURACION']],
              left_on='RUBRO',
              right_on='RUBRO',
              how='left'
       )

       daily_resource_usage_list = []

       for index, row in recursos_tareas_df.iterrows():
              task_id = row['IDRUBRO']
              resource_name = row['RECURSO']
              unit = row['UNIDAD']
              total_quantity = row['CANTIDAD']
              start_date = row['FECHAINICIO']
              end_date = row['FECHAFIN']
              duration_days = row['DURACION']

              if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
                     st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea ID {task_id}, recurso '{resource_name}'. Saltando.")
                     continue
    
              if duration_days <= 0:
                   daily_quantity = total_quantity
                   date_range = [start_date]
              else:
                   daily_quantity = total_quantity / (duration_days + 1)
                   date_range = pd.date_range(start=start_date, end=end_date, freq='D')
       
              temp_df = pd.DataFrame({
                   'Fecha': date_range,
                   'IDRUBRO': task_id,
                   'RECURSO': resource_name,
                   'UNIDAD': unit,
                   'Cantidad_Diaria': daily_quantity,
                   'Cantidad_Total_Tarea': total_quantity
               })
               
              daily_resource_usage_list.append(temp_df)

       if daily_resource_usage_list:
              all_daily_resource_usage_df = pd.concat(daily_resource_usage_list, ignore_index=True)
       else:
              st.warning("\nNo se generaron datos de uso diario de recursos.")
              all_daily_resource_usage_df = pd.DataFrame()
        
       daily_resource_demand_df = all_daily_resource_usage_df.groupby(
              ['Fecha', 'RECURSO', 'UNIDAD'],
              as_index=False
       )['Cantidad_Diaria'].sum()

       daily_resource_demand_df.rename(columns={'Cantidad_Diaria': 'Demanda_Diaria_Total'}, inplace=True)
       daily_resource_demand_df['RECURSO'] = daily_resource_demand_df['RECURSO'].str.strip()
       recursos_df['RECURSO'] = recursos_df['RECURSO'].str.strip()
    
       resource_demand_with_details_df = daily_resource_demand_df.merge(
              recursos_df[['RECURSO', 'TYPE', 'TARIFA']],
              on='RECURSO',
              how='left'
       )
       resource_demand_with_details_df['Costo_Diario'] = resource_demand_with_details_df['Demanda_Diaria_Total'] * resource_demand_with_details_df['TARIFA']

       daily_cost_by_type_df = resource_demand_with_details_df.groupby(
              ['Fecha', 'TYPE'],
              as_index=False
       )['Costo_Diario'].sum()

       daily_demand_by_resource_df = resource_demand_with_details_df.groupby(
              ['Fecha', 'RECURSO', 'UNIDAD'],
              as_index=False
       )['Demanda_Diaria_Total'].sum()

       with tab3:

           import pandas as pd
           import plotly.graph_objects as go
           import plotly.express as px
           import re
           from datetime import timedelta, datetime
           from collections import defaultdict
           st.subheader("📊 Distribución de Recursos")
       
           if 'RUBRO' not in recursos_tareas_df.columns:
       
               if 'tareas_df' in locals() or 'tareas_df' in globals():
       
                   tareas_df['RUBRO'] = tareas_df['RUBRO'].astype(str).str.strip()
                   recursos_tareas_df['RUBRO'] = recursos_tareas_df['RUBRO'].astype(str).str.strip()
           
                   if 'IDRUBRO' in recursos_tareas_df.columns and 'IDRUBRO' in tareas_df.columns:
                        recursos_tareas_df = recursos_tareas_df.merge(
                           tareas_df[['IDRUBRO', 'RUBRO']],
                           left_on='IDRUBRO',
                           right_on='IDRUBRO',
                           how='left'
                       )
                        st.warning("Re-merged to include 'RUBRO' column using IDRUBRO.")
                   else:
                        st.warning("❌ Error: 'IDRUBRO' column not found in one of the dataframes. Cannot re-add 'RUBRO'.")
                        raise KeyError("'IDRUBRO' column not found for re-merging.")
           
               else:
                   st.warning("❌ Error: 'tareas_df' not found. Cannot re-add 'RUBRO' column.")
                   raise NameError("'tareas_df' not found.")
           
           unique_rubros = sorted(recursos_tareas_df['RUBRO'].dropna().unique().tolist())
       
           fig_resource_timeline = go.Figure()
       
           pastel_blue = 'rgb(174, 198, 207)' 
       
           for i, row in recursos_tareas_df.iterrows():
               fig_resource_timeline.add_trace(go.Scattergl(
                   x=[row['FECHAINICIO'], row['FECHAFIN']],
                   y=[row['RECURSO'], row['RECURSO']],
                   mode='lines',
                   line=dict(color=pastel_blue, width=10), 
                   name=row['RECURSO'], 
                   showlegend=False, 
                   hoverinfo='text',
                   text=f"<b>Rubro:</b> {row['RUBRO']}<br><b>Recurso:</b> {row['RECURSO']}<br><b>Inicio:</b> {row['FECHAINICIO'].strftime('%Y-%m-%d')}<br><b>Fin:</b> {row['FECHAFIN'].strftime('%Y-%m-%d')}",
                   customdata=[row['RUBRO']] 
               ))
       
           dropdown_options = [{'label': 'All Tasks', 'method': 'update', 'args': [{'visible': [True] * len(fig_resource_timeline.data)}, {'title': 'Línea de Tiempo de Uso de Recursos'}]}]
       
           for rubro in unique_rubros:
       
               visibility = [trace.customdata[0] == rubro for trace in fig_resource_timeline.data if trace.customdata and trace.customdata[0] in unique_rubros]
       
               visibility_list = [trace.customdata[0] == rubro if trace.customdata and len(trace.customdata) > 0 else False for trace in fig_resource_timeline.data]
           
               dropdown_options.append({
                   'label': rubro,
                   'method': 'update',
                   'args': [{'visible': visibility_list}, {'title': f'Línea de Tiempo de Uso de Recursos (Filtrado por: {rubro})'}]
               })
       
           fig_resource_timeline.update_layout(
               updatemenus=[
                   go.layout.Updatemenu(
                       buttons=dropdown_options,
                       direction="down",
                       pad={"r": 10, "t": 10},
                       showactive=True,
                       x=0.01,
                       xanchor="left",
                       y=1.1,
                       yanchor="top"
                   ),
               ],
               yaxis=dict(
                   autorange="reversed",
                   title="Recurso",
                   tickfont=dict(size=10) 
               ),
               xaxis=dict(
                   title='Fechas',
                   side='bottom',
                   dtick='M1', 
                   tickangle=-90, 
                   showgrid=True,
                   gridcolor='rgba(128,128,128,0.3)',
                   gridwidth=0.5
               ),
               xaxis2=dict(
                   title='Fechas',
                   overlaying='x',
                   side='top',
                   dtick='M1',
                   tickangle=90,
                   showgrid=True,
                   gridcolor='rgba(128,128,128,0.3)',
                   gridwidth=0.5
               ),
               height=max(600, len(recursos_tareas_df['RECURSO'].unique()) * 20), 
               showlegend=False,
               plot_bgcolor='white',
               hovermode='closest'
           )
           st.plotly_chart(fig_resource_timeline, use_container_width=True)
    #__________________________________________________________________________________________________

       if 'resource_demand_with_details_df' in locals() or 'resource_demand_with_details_df' in globals():
              required_columns_and_types = {
                     'Fecha': 'datetime64[ns]',
                     'RECURSO': 'object', 
                     'UNIDAD': 'object', 
                     'Demanda_Diaria_Total': 'float64', 
                     'TYPE': 'object',
                     'TARIFA': 'float64', 
                     'Costo_Diario': 'float64' 
              }
              missing_columns = [col for col in required_columns_and_types if col not in resource_demand_with_details_df.columns]
              if not missing_columns:
                     type_issues = []
                     for col, expected_type in required_columns_and_types.items():
                            if expected_type == 'object':
                                   if pd.api.types.is_numeric_dtype(resource_demand_with_details_df[col]):
                                          type_issues.append(f"Column '{col}' is numeric but expected object (string).")
                            elif not pd.api.types.is_dtype_equal(resource_demand_with_details_df[col].dtype, expected_type):
                                   if expected_type == 'float64' and pd.api.types.is_integer_dtype(resource_demand_with_details_df[col]):
                                          pass 
                                   else:
                                          type_issues.append(f"Column '{col}' has type {resource_demand_with_details_df[col].dtype} but expected {expected_type}.")
              else:
                     st.warning(f"❌ Error: Missing required columns in resource_demand_with_details_df: {missing_columns}")
       else:
              st.warning("❌ Error: DataFrame 'resource_demand_with_details_df' not found.")

       resource_demand_with_details_df['Fecha'] = pd.to_datetime(resource_demand_with_details_df['Fecha'])
       resource_demand_with_details_df['Periodo_Mensual'] = resource_demand_with_details_df['Fecha'].dt.to_period('M')
       monthly_costs_df = resource_demand_with_details_df.groupby('Periodo_Mensual')['Costo_Diario'].sum().reset_index()
       monthly_costs_df['Periodo_Mensual'] = monthly_costs_df['Periodo_Mensual'].astype(str) 
       monthly_costs_df['Costo_Acumulado'] = monthly_costs_df['Costo_Diario'].cumsum()
     
       def format_currency(value):
              if pd.notna(value):
                     return f"S/ {value:,.2f}"  
              return "S/ 0.00"  
       
       monthly_costs_df['Costo_Mensual_Formateado'] = monthly_costs_df['Costo_Diario'].apply(format_currency)
       monthly_costs_df['Costo_Acumulado_Formateado'] = monthly_costs_df['Costo_Acumulado'].apply(format_currency)

       with tab4:
           from plotly.subplots import make_subplots
           import plotly.graph_objects as go
           
           st.subheader("📊 Cronograma Valorado")
           
           fig = make_subplots(specs=[[{"secondary_y": True}]])
           
           # Barra de Costo Mensual
           fig.add_bar(
               x=monthly_costs_df['Periodo_Mensual'],
               y=monthly_costs_df['Costo_Diario'],
               name='Costo Mensual',
               text=monthly_costs_df['Costo_Mensual_Formateado'],
               hoverinfo='text',
               hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
               secondary_y=False
           )
           
           # Línea de Costo Acumulado
           fig.add_scatter(
               x=monthly_costs_df['Periodo_Mensual'],
               y=monthly_costs_df['Costo_Acumulado'],
               mode='lines+markers',
               name='Costo Acumulado',
               text=monthly_costs_df['Costo_Acumulado_Formateado'],
               hoverinfo='text',
               hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
               line=dict(color='red'),
               secondary_y=True
           )
           
           # Configuración de ejes
           fig.update_yaxes(
               title_text="Costo Mensual",
               secondary_y=False,
               showgrid=False,  # ❌ Sin grilla horizontal para la barra
               range=[0, monthly_costs_df['Costo_Diario'].max()*1.1]
           )
           fig.update_yaxes(
               title_text="Costo Acumulado",
               secondary_y=True,
               showgrid=True,   # ✅ Solo el acumulador tiene grilla
               gridcolor='lightgrey',
               range=[0, monthly_costs_df['Costo_Acumulado'].max()*1.1]
           )
           fig.update_xaxes(title_text="Período Mensual", tickangle=-45)
           
           # Layout
           fig.update_layout(
               hovermode='x unified',
               height=600,
               legend=dict(
                   x=1.1,
                   y=1,
                   bgcolor='rgba(255, 255, 255, 0.5)',
                   bordercolor='rgba(0, 0, 0, 0.5)'
               ),
               plot_bgcolor='white'
           )
           
           st.plotly_chart(fig, use_container_width=True)


else:
       st.warning("Sube el archivo Excel con las hojas Tareas, Recursos y Dependencias.")



























































