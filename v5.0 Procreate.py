import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict, deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")
st.title("📊 Gestión de Proyectos - Cronograma Valorado y Recursos")

# --- Subir archivo ---
archivo_excel = st.file_uploader("Subir archivo Excel con hojas Tareas, Recursos y Dependencias", type=["xlsx"])

if archivo_excel:
    try:
        tareas_df = pd.read_excel(archivo_excel, sheet_name='Tareas')
        recursos_df = pd.read_excel(archivo_excel, sheet_name='Recursos')
        dependencias_df = pd.read_excel(archivo_excel, sheet_name='Dependencias')
    except:
        st.error("El archivo debe contener las hojas: Tareas, Recursos y Dependencias")
        st.stop()

    # --- Mostrar tablas editables ---
    st.subheader("📋 Tabla Tareas")
    gb = GridOptionsBuilder.from_dataframe(tareas_df)
    gb.configure_default_column(editable=True)
    tareas_grid = AgGrid(tareas_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    tareas_df = tareas_grid['data']

    st.subheader("📋 Tabla Recursos")
    gb = GridOptionsBuilder.from_dataframe(recursos_df)
    gb.configure_default_column(editable=True)
    recursos_grid = AgGrid(recursos_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    recursos_df = recursos_grid['data']

    st.subheader("📋 Tabla Dependencias")
    gb = GridOptionsBuilder.from_dataframe(dependencias_df)
    gb.configure_default_column(editable=True)
    dependencias_grid = AgGrid(dependencias_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    dependencias_df = dependencias_grid['data']

    # Transformar directamente en las columnas FECHAINICIO y FECHAFIN
    for col in ['FECHAINICIO','FECHAFIN']:
        # Convertir de string ISO a datetime
        tareas_df[col] = pd.to_datetime(tareas_df[col], errors='coerce')
        # Formatear como DD/MM/YYYY
        tareas_df[col] = tareas_df[col].dt.strftime('%d/%m/%Y')
    
    # Transformar las columnas de texto DD/MM/YYYY a datetime
    for col in ['FECHAINICIO','FECHAFIN']:
        tareas_df[col] = pd.to_datetime(tareas_df[col], dayfirst=True, errors='coerce')

    # --- Calcular duración ---
    tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days
    tareas_df.loc[tareas_df['DURACION'] < 0, 'DURACION'] = 0  # prevenir negativos

    # --- Predecesoras ---
    tareas_df['PREDECESORAS'] = tareas_df['PREDECESORAS'].fillna('').astype(str)

    # --- Tarifas ---
    if 'TARIFA' in recursos_df.columns:
        recursos_df['TARIFA'] = pd.to_numeric(recursos_df['TARIFA'], errors='coerce').fillna(0)

    #funcion de clacular fechas
    import pandas as pd
    from datetime import timedelta
    import re
    from collections import defaultdict, deque
    
    def calcular_fechas(df):
        df = df.copy()
        df.columns = df.columns.str.strip()  # Limpiar espacios
    
        # Diccionarios de inicio, fin y duración
        inicio_rubro = df.set_index('IDRUBRO')['FECHAINICIO'].to_dict()
        fin_rubro = df.set_index('IDRUBRO')['FECHAFIN'].to_dict()
        duracion_rubro = (df.set_index('IDRUBRO')['FECHAFIN'] - df.set_index('IDRUBRO')['FECHAINICIO']).dt.days.to_dict()
    
        # Construir grafo de dependencias y contar predecesores
        dependencias = defaultdict(list)
        pre_count = defaultdict(int)
        for _, row in df.iterrows():
            tarea_id = row['IDRUBRO']
            predecesoras_str = str(row['PREDECESORAS']).strip()
            if predecesoras_str not in ['nan','']:
                pre_list = predecesoras_str.split(',')
                for pre in pre_list:
                    pre = pre.strip()
                    match = re.match(r'(\d+)', pre)
                    if match:
                        pre_id = int(match.group(1))
                        dependencias[pre_id].append(tarea_id)
                        pre_count[tarea_id] += 1
    
        # Inicializar cola con tareas sin predecesores
        queue = deque([tid for tid in df['IDRUBRO'] if pre_count[tid] == 0])
    
        # Diccionarios para fechas calculadas
        inicio_calc = inicio_rubro.copy()
        fin_calc = fin_rubro.copy()
    
        # Procesar tareas en orden topológico
        while queue:
            tarea_id = queue.popleft()
            row = df[df['IDRUBRO']==tarea_id].iloc[0]
            duracion = duracion_rubro[tarea_id]
            predecesoras_str = str(row['PREDECESORAS']).strip()
    
            nueva_inicio = inicio_calc[tarea_id]
            nueva_fin = fin_calc[tarea_id]
    
            if predecesoras_str not in ['nan','']:
                pre_list = predecesoras_str.split(',')
                for pre in pre_list:
                    pre = pre.strip()
                    match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre)
                    if match:
                        pre_id = int(match.group(1))
                        tipo = match.group(2).upper() if match.group(2) else 'FC'
                        desfase = int(match.group(3)) if match.group(3) else 0
    
                        if pre_id in inicio_calc and pre_id in fin_calc:
                            inicio_pre = inicio_calc[pre_id]
                            fin_pre = fin_calc[pre_id]
    
                            # --- Ajuste según tipo de dependencia ---
                            if tipo == 'CC':
                                nueva_inicio = inicio_pre + timedelta(days=desfase)
                                nueva_fin = nueva_inicio + timedelta(days=duracion)
                            elif tipo == 'FC':
                                nueva_inicio = fin_pre + timedelta(days=desfase)
                                nueva_fin = nueva_inicio + timedelta(days=duracion)
                            elif tipo == 'CF':
                                nueva_fin = inicio_pre + timedelta(days=desfase)
                                nueva_inicio = nueva_fin - timedelta(days=duracion)
                            elif tipo == 'FF':
                                nueva_fin = fin_pre + timedelta(days=desfase)
                                nueva_inicio = nueva_fin - timedelta(days=duracion)
                            else:
                                print(f"⚠️ Tipo de relación '{tipo}' no reconocido en '{pre}' para tarea {tarea_id}")
    
                            print(f"Tarea {tarea_id} - Predecesor {pre_id} ({tipo}{desfase:+}): "
                                  f"Inicio pre {inicio_pre}, Fin pre {fin_pre} -> "
                                  f"Inicio calculado {nueva_inicio}, Fin calculado {nueva_fin}")
    
            # Actualizar fechas calculadas
            inicio_calc[tarea_id] = nueva_inicio
            fin_calc[tarea_id] = nueva_fin
    
            # Agregar hijos a la cola
            for hijo in dependencias[tarea_id]:
                pre_count[hijo] -= 1
                if pre_count[hijo] == 0:
                    queue.append(hijo)
    
        # Actualizar DataFrame
        df['FECHAINICIO'] = df['IDRUBRO'].map(inicio_calc)
        df['FECHAFIN'] = df['IDRUBRO'].map(fin_calc)
    
        return df

    tareas_df = calcular_fechas(tareas_df)

    # _________________________________________________________________________________________________

        # Import necessary libraries if not already imported in this cell's execution conext
    # Although imports are in other cells, ensuring them here makes the cell self-contained
    import pandas as pd
    from datetime import timedelta, datetime
    import re
    from collections import defaultdict, deque
    import math 
    
    try:

        if 'tareas_df' not in locals() and 'tareas_df' not in globals():
             raise NameError("tareas_df not found, attempting to load.")

        if not pd.api.types.is_datetime64_any_dtype(tareas_df['FECHAINICIO']) or \
           not pd.api.types.is_datetime64_any_dtype(tareas_df['FECHAFIN']):
           print("Date columns not in datetime format, re-converting.")
           tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'], dayfirst=True)
           tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'], dayfirst=True)
    
    except (NameError, KeyError) as e:
        st.warning(f"Error checking tareas_df ({e}). Attempting to reload and process dates.")

        if 'archivo' in locals() or 'archivo' in globals():
            try:
                tareas_df = pd.read_excel(archivo, sheet_name='Tareas')
                tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'], dayfirst=True)
                tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'], dayfirst=True)
                st.warning("tareas_df re-loaded and dates converted.")
            except Exception as load_error:
                st.warning(f"❌ Error re-loading tareas_df: {load_error}")
                raise load_error # Re-raise the error if loading fails
        else:
            st.warning("❌ Error: 'archivo' variable not found. Cannot re-load tareas_df.")
            raise NameError("'archivo' variable not found. Cannot proceed.") # Stop execution
    tareas_df.columns = tareas_df.columns.str.strip()

    tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days.fillna(0).astype(int)

    es = {} # Early Start (datetime)
    ef = {} # Early Finish (datetime)
    ls = {} # Late Start (datetime)
    lf = {} # Late Finish (datetime)
    tf = {} # Total Float (timedelta)
    ff = {} # Free Float (timedelta)
    duracion_dict = tareas_df.set_index('IDRUBRO')['DURACION'].to_dict()
    
    # Build dependency graph (predecessors to successors) and reverse graph
    dependencias = defaultdict(list) # pre_id -> [suc_id1, suc_id2, ...]
    predecesoras_map = defaultdict(list) # suc_id -> [(pre_id1, type1, lag1), (pre_id2, type2, lag2), ...]
    all_task_ids = set(tareas_df['IDRUBRO'].tolist())
    
    for _, row in tareas_df.iterrows():
        tarea_id = row['IDRUBRO']
        predecesoras_str = str(row['PREDECESORAS']).strip()
    
        if predecesoras_str not in ['nan', '']:
            pre_list = predecesoras_str.split(',')
            for pre_entry in pre_list:
                pre_entry = pre_entry.strip()
                # Expresión regular para capturar ID, Tipo (opcional, 2 letras), Desfase (opcional, con signo)
                match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
    
                if match:
                    pre_id = int(match.group(1))
                    tipo_relacion = match.group(2).upper() if match.group(2) else 'FC' # Default to FC
                    desfase = int(match.group(3)) if match.group(3) else 0 # Default to 0 lag
    
                    # Only add dependency if predecessor ID exists in the tasks list
                    if pre_id in all_task_ids:
                        dependencias[pre_id].append(tarea_id)
                        predecesoras_map[tarea_id].append((pre_id, tipo_relacion, desfase))
                    else:
                         st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} mencionado en '{pre_entry}' para tarea {tarea_id} no encontrado en la lista de tareas. Ignorando esta dependencia.")
                else:
                    if pre_entry != '': # Avoid warning for empty strings
                        st.warning(f"⚠️ Advertencia: Formato de predecesora '{pre_entry}' no reconocido para la tarea {tarea_id}. Ignorando.")

    initial_tasks_ids = [tid for tid in all_task_ids if tid not in predecesoras_map]

    for tid in initial_tasks_ids:
         task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
         if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
            es[tid] = task_row.iloc[0]['FECHAINICIO']
            # Ensure duration is available and is a number
            duration = duracion_dict.get(tid, 0)
            if not isinstance(duration, (int, float)): duration = 0
            ef[tid] = es[tid] + timedelta(days=duration)
         else:
             st.warning(f"⚠️ Advertencia: Tarea ID {tid} no encontrada o FECHAINICIO inválida para inicializar ES/EF.")

    queue = deque(initial_tasks_ids)
    processed_forward = set(initial_tasks_ids)
    
    # Keep track of how many predecessors of each task have been processed
    predecessor_process_count = defaultdict(int)
    in_degree = {tid: len(predecesoras_map.get(tid, [])) for tid in all_task_ids} # Count incoming dependencies
    
    # Initialize queue with tasks having 0 incoming dependencies based on parsed map
    queue = deque([tid for tid in all_task_ids if in_degree[tid] == 0])
    processed_forward = set(queue) # Tasks initially in queue are processed w.r.t. dependencies
    
    # Initialize ES/EF for tasks with 0 in_degree based on their original start date
    for tid in queue:
        task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
        if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
            es[tid] = task_row.iloc[0]['FECHAINICIO']
            duration = duracion_dict.get(tid, 0)
            if not isinstance(duration, (int, float)): duration = 0
            ef[tid] = es[tid] + timedelta(days=duration)
        else:
            st.warning(f"⚠️ Advertencia: Tarea ID {tid} (0 in-degree) no encontrada o FECHAINICIO inválida para inicializar ES/EF.")

            processed_forward.discard(tid) # Remove from processed if initialization failed
            if tid in es: del es[tid]
            if tid in ef: del ef[tid]
    
    while queue:
        u = queue.popleft() # Current task being processed
        for v in dependencias.get(u, []):
            for pre_id_v, tipo_v, desfase_v in predecesoras_map.get(v, []):
                if pre_id_v == u: # Found the dependency from u to v
                    # Calculate potential Early Start for v based on u's dates and the dependency
                    potential_es_v = None
                    duration_v = duracion_dict.get(v, 0)
                    if not isinstance(duration_v, (int, float)): duration_v = 0
    
                    if u in ef and u in es:
                        if tipo_v == 'CC': # Start-to-Start
                            potential_es_v = es[u] + timedelta(days=desfase_v)
                        elif tipo_v == 'FC': # Finish-to-Start (Most common)
                            potential_es_v = ef[u] + timedelta(days=desfase_v)
                        elif tipo_v == 'CF': # Start-to-Finish
                             potential_es_v = (es[u] + timedelta(days=desfase_v)) - timedelta(days=duration_v)
    
                        elif tipo_v == 'FF': # Finish-to-Finish
                             potential_es_v = (ef[u] + timedelta(days=desfase_v)) - timedelta(days=duration_v)
                        else:
                             st.warning(f"⚠️ Tipo de relación '{tipo_v}' no reconocido para calcular ES de tarea {v} basada en {u}. Usando lógica FC por defecto.")
                             potential_es_v = ef[u] + timedelta(days=desfase_v) # Default to FC logic

                        if v not in es or (potential_es_v is not None and potential_es_v > es[v]):
                            es[v] = potential_es_v

                        if v in es: # Ensure ES[v] was set
                            duration_v_calc = duracion_dict.get(v, 0)
                            if not isinstance(duration_v_calc, (int, float)): duration_v_calc = 0
                            ef[v] = es[v] + timedelta(days=duration_v_calc)
    
                    else:
                        st.warning(f"⚠️ Advertencia: ES/EF no calculados para predecesor ID {u} al procesar sucesor ID {v}. Saltando cálculo de ES/EF para v basado en u.")

            in_degree[v] -= 1

            if in_degree[v] == 0 and v not in processed_forward:
                 queue.append(v)
                 processed_forward.add(v)
    
    # Check if all tasks were processed in forward pass (indicates potential cycles or disconnected graph)
    unprocessed_forward = all_task_ids - processed_forward
    if unprocessed_forward:
        st.warning(f"⚠️ Advertencia: Las siguientes tareas no fueron procesadas en el pase hacia adelante (posible ciclo o grafo desconectado): {unprocessed_forward}")

        for tid in unprocessed_forward:
             if tid not in es:
                 task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
                 if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
                     es[tid] = task_row.iloc[0]['FECHAINICIO']
                     duration = duracion_dict.get(tid, 0)
                     if not isinstance(duration, (int, float)): duration = 0
                     ef[tid] = es[tid] + timedelta(days=duration)
                     st.warning(f"Inicializando ES/EF para tarea no procesada {tid} con su fecha de inicio original.")
                 else:
                     st.warning(f"❌ Error: Tarea no procesada {tid} no encontrada o FECHAINICIO inválida. No se pudo inicializar ES/EF.")
    

    end_tasks_ids = [tid for tid in all_task_ids if tid not in dependencias]

    project_finish_date = None
    if ef: # Check if ef dictionary is not empty
        project_finish_date = max(ef.values())
    else:
        st.warning("❌ Error: No se calculó ninguna Fecha de Finalización Temprana (EF) en el pase hacia adelante. No se puede determinar la fecha de fin del proyecto.")

        raise ValueError("No EF calculated in forward pass.")

    for tid in end_tasks_ids:
        if tid in ef: # Ensure EF was calculated for the end task in the forward pass
            lf[tid] = project_finish_date
            duration = duracion_dict.get(tid, 0)
            if not isinstance(duration, (int, float)): duration = 0
            ls[tid] = lf[tid] - timedelta(days=duration)
        else:
            if tid in ef:
                 lf[tid] = ef[tid] # Assume LF = EF if not processed fully forward
                 duration = duracion_dict.get(tid, 0)
                 if not isinstance(duration, (int, float)): duration = 0
                 ls[tid] = lf[tid] - timedelta(days=duration)
                 st.warning(f"⚠️ Advertencia: Tarea final ID {tid} no procesada completamente hacia adelante. Inicializando LF/LS basado en su EF.")
            else:
                st.warning(f"⚠️ Advertencia: Tarea final ID {tid} no encontrada en EF. No se puede inicializar LF/LS.")

    queue_backward = deque(end_tasks_ids)
    processed_backward = set(end_tasks_ids)

    successor_process_count = defaultdict(int)

    successor_details_map = defaultdict(list)
    for suc_id, pre_list in predecesoras_map.items():
        for pre_id, type_rel, lag in pre_list:
            successor_details_map[pre_id].append((suc_id, type_rel, lag))

    while queue_backward:
        v = queue_backward.popleft() # Current task being processed (working backwards, v is a successor)

        for u, tipo_relacion_uv, desfase_uv in predecesoras_map.get(v, []):
            # Calculate potential Late Finish for predecessor u based on v's dates and the dependency u -> v
            potential_lf_u = None
    
            # Ensure v's LS/LF are available before calculating potential_lf_u
            if v in ls and v in lf:
                if tipo_relacion_uv == 'CC': # Start-to-Start: LS(v) = ES(u) + lag => ES(u) = LS(v) - lag. LF(u) = ES(u) + Duration(u)

                    potential_lf_u = (ls[v] - timedelta(days=desfase_uv)) + timedelta(days=duracion_dict.get(u, 0))
    
                elif tipo_relacion_uv == 'FC': # Finish-to-Start: LS(v) = EF(u) + lag => EF(u) = LS(v) - lag. LF(u) = EF(u)
                     # LF(u) must be <= LS(v) - lag
                     potential_lf_u = ls[v] - timedelta(days=desfase_uv)
    
                elif tipo_relacion_uv == 'CF': # Start-to-Finish: LF(v) = ES(u) + lag => ES(u) = LF(v) - lag. LF(u) = ES(u) + Duration(u)
                     # This dependency constrains u's ES based on v's LF.
                     # LF(u) must be <= (LF(v) - lag) + Duration(u)
                     potential_lf_u = (lf[v] - timedelta(days=desfase_uv)) + timedelta(days=duracion_dict.get(u, 0))
    
                elif tipo_relacion_uv == 'FF': # Finish-to-Finish: LF(v) = EF(u) + lag => EF(u) = LF(v) - lag. LF(u) = EF(u)
                     # LF(u) must be <= LF(v) - lag
                     potential_lf_u = lf[v] - timedelta(days=desfase_uv)
    
                else:
                    st.warning(f"⚠️ Tipo de relación '{tipo_relacion_uv}' no reconocido para calcular LF de tarea {u} basada en {v}. Usando lógica FC por defecto.")
                    potential_lf_u = ls[v] - timedelta(days=desfase_uv) # Default to FC logic
    
    
                # Update LF[u] if the calculated potential LF is earlier than the current LF[u]
                # Initialize LF[u] with a very late date (infinity) if it doesn't exist yet
                # Use a date far in the future instead of infinity for datetime comparisons
                if u not in lf or (potential_lf_u is not None and potential_lf_u < lf[u]):
                    lf[u] = potential_lf_u
                    # Once LF[u] is updated, update LS[u]
                    duration_u = duracion_dict.get(u, 0)
                    if not isinstance(duration_u, (int, float)): duration = 0
                    ls[u] = lf[u] - timedelta(days=duration_u)
    
            else:
                 st.warning(f"⚠️ Advertencia: LS/LF no calculados para sucesora ID {v} al procesar predecesora ID {u}. Saltando cálculo de LF/LS para u basado en v.")
    

            total_successors_of_u = len(dependencias.get(u, []))
            # Increment count for u because one of its successors (v) has just been processed backward
            successor_process_count[u] += 1
    
            # Add u to queue_backward if all its successors have been processed in the backward pass
            if successor_process_count[u] == total_successors_of_u and u not in processed_backward:
                 queue_backward.append(u)
                 processed_backward.add(u)
    
    
    # Check if all tasks were processed in backward pass
    unprocessed_backward = all_task_ids - processed_backward
    if unprocessed_backward:
        st.warning(f"⚠️ Advertencia: Las siguientes tareas no fueron procesadas en el pase hacia atrás (posible ciclo o problema en el grafo/inicialización): {unprocessed_backward}")

        for tid in unprocessed_backward:
            if tid in es and tid in ef:
                lf[tid] = ef[tid]
                ls[tid] = es[tid]
                st.warning(f"Inicializando LF/LS para tarea no procesada hacia atrás {tid} con sus fechas tempranas.")
            else:
                st.warning(f"❌ Error: Tarea no procesada hacia atrás {tid} no encontrada en ES/EF. No se pudo inicializar LF/LS.")

    for tid in all_task_ids:
        if tid in ef and tid in lf:
            # TF = LF - EF or LS - ES
            tf[tid] = lf[tid] - ef[tid]
            # Ensure TF is not negative due to floating point issues, set to 0 if very close
            if tf[tid].total_seconds() < -1e-9:
                 tf[tid] = timedelta(days=0)
    
            # Calculate Free Float (FF = min(ES of successors) - EF of current task)
            min_successor_es = None
            for suc_id in dependencias.get(tid, []):
                # Find the specific dependency relationship from tid to suc_id
                 for pre_id_suc, tipo_suc, desfase_suc in predecesoras_map.get(suc_id, []):
                     if pre_id_suc == tid: # Found the dependency tid -> suc_id
                         # Calculate the required start time for suc_id based on tid and the dependency
                         required_start_suc = None
                         if tid in es and tid in ef: # Ensure tid's dates are available
                             if tipo_suc == 'CC': # Start-to-Start: required ES(suc) = ES(tid) + lag
                                  required_start_suc = es[tid] + timedelta(days=desfase_suc)
                             elif tipo_suc == 'FC': # Finish-to-Start: required ES(suc) = EF(tid) + lag
                                  required_start_suc = ef[tid] + timedelta(days=desfase_suc)
                             elif tipo_suc == 'CF': # Start-to-Finish: required LF(suc) = ES(tid) + lag => required ES(suc) = (ES(tid) + lag) - Duration(suc)
                                  duration_suc = duracion_dict.get(suc_id, 0)
                                  if not isinstance(duration_suc, (int, float)): duration_suc = 0
                                  required_start_suc = (es[tid] + timedelta(days=desfase_suc)) - timedelta(days=duration_suc)
                             elif tipo_suc == 'FF': # Finish-to-Finish: required LF(suc) = EF(tid) + lag => required ES(suc) = (EF(tid) + lag) - Duration(suc)
                                  duration_suc = duracion_dict.get(suc_id, 0)
                                  if not isinstance(duration_suc, (int, float)): duration_suc = 0
                                  required_start_suc = (ef[tid] + timedelta(days=desfase_suc)) - timedelta(days=duration_suc)
                             else:
                                  # Should not happen if parsing was correct, but as fallback
                                  st.warning(f"⚠️ Tipo de relación '{tipo_suc}' no reconocido al calcular FF para tarea {tid} basada en sucesor {suc_id}. Usando lógica FC por defecto.")
                                  required_start_suc = ef[tid] + timedelta(days=desfase_suc) # Default to FC logic
    
                             # Update min_successor_es
                             if required_start_suc is not None:
                                 if min_successor_es is None or required_start_suc < min_successor_es:
                                     min_successor_es = required_start_suc
    
                             break # Found the dependency details for this successor, move to next successor
    
            if min_successor_es is not None and tid in ef:
                 ff[tid] = min_successor_es - ef[tid]
                 # Ensure FF is not negative
                 if ff[tid].total_seconds() < -1e-9:
                      ff[tid] = timedelta(days=0)
            else:
                 ff[tid] = timedelta(days=0) # If no successors, FF is 0
    
        else:
            # If ES/EF or LF/LS are missing, TF/FF cannot be calculated.
            tf[tid] = pd.NA # Assign missing indicator
            ff[tid] = pd.NA # Assign missing indicator


    tareas_df['FECHA_INICIO_TEMPRANA'] = tareas_df['IDRUBRO'].map(es)
    tareas_df['FECHA_FIN_TEMPRANA'] = tareas_df['IDRUBRO'].map(ef)
    tareas_df['FECHA_INICIO_TARDE'] = tareas_df['IDRUBRO'].map(ls)
    tareas_df['FECHA_FIN_TARDE'] = tareas_df['IDRUBRO'].map(lf)
    tareas_df['HOLGURA_TOTAL_TD'] = tareas_df['IDRUBRO'].map(tf) # Store as timedelta first
    tareas_df['HOLGURA_LIBRE_TD'] = tareas_df['IDRUBRO'].map(ff) # Store as timedelta first

    tareas_df['HOLGURA_TOTAL'] = tareas_df['HOLGURA_TOTAL_TD'].apply(lambda x: x.days if pd.notna(x) else pd.NA)
    tareas_df['HOLGURA_LIBRE'] = tareas_df['HOLGURA_LIBRE_TD'].apply(lambda x: x.days if pd.notna(x) else pd.NA)
    
    tolerance_days = 1e-9
    tareas_df['RUTA_CRITICA'] = tareas_df['HOLGURA_TOTAL'].apply(lambda x: abs(x) < tolerance_days if pd.notna(x) else False)


    # _________________________________________________________________________________________________
    


    # --- Mostrar tabla ---
    st.subheader("📋 Tareas con Fechas Calculadas y Ruta Crítica")
    st.dataframe(tareas_df[['IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                            'FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA',
                            'FECHA_INICIO_TARDE','FECHA_FIN_TARDE','DURACION','HOLGURA_TOTAL','RUTA_CRITICA']])

    dependencias_df = dependencias_df.merge(recursos_df, left_on='RECURSO', right_on='RECURSO', how='left')
    dependencias_df['COSTO'] = dependencias_df['CANTIDAD'] * dependencias_df['TARIFA']

    costos_por_can = dependencias_df.groupby('RUBRO', as_index=False)['COSTO'].sum()
    costos_por_can.rename(columns={'RUBRO': 'RUBRO', 'COSTO': 'COSTO_TOTAL'}, inplace=True)
    
    tareas_df['RUBRO'] = tareas_df['RUBRO'].str.strip()
    costos_por_can['RUBRO'] = costos_por_can['RUBRO'].str.strip()

    tareas_df = tareas_df.merge(costos_por_can[['RUBRO', 'COSTO_TOTAL']], on='RUBRO', how='left')
    
    # --- Diagrama de Gantt ---
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import re
    from datetime import timedelta, datetime
    st.subheader("📊 Diagrama de Gantt - Ruta Crítica")
    
    cost_column_name = None
    if 'COSTO_TOTAL_RUBRO' in tareas_df.columns:
        cost_column_name = 'COSTO_TOTAL_RUBRO'
    elif 'COSTO_TOTAL_x' in tareas_df.columns: # Fallback if the merge created this
         cost_column_name = 'COSTO_TOTAL_x'
    elif 'COSTO_TOTAL' in tareas_df.columns: # Fallback for the originally calculated column
         cost_column_name = 'COSTO_TOTAL'
    
    if cost_column_name:
        # Convertir a numérico si aún no lo está. Usar errors='coerce' para convertir no-numéricos a NaN.
        tareas_df[cost_column_name] = pd.to_numeric(tareas_df[cost_column_name], errors='coerce')
        # Llenar NaN con 0 si es necesario para el cálculo o visualización
        tareas_df[cost_column_name] = tareas_df[cost_column_name].fillna(0)
    else:
        print("⚠️ Advertencia: No se encontró una columna de costos reconocida en el DataFrame.")
        # Crear una columna de costo con 0s si no se encontró ninguna
        tareas_df['COSTO_TOTAL_NUMERICO'] = 0
        cost_column_name = 'COSTO_TOTAL_NUMERICO'
    
    if 'IDRUBRO' in tareas_df.columns:
        # Ordenar por IDRUBRO de forma ascendente
        tareas_df = tareas_df.sort_values(['IDRUBRO'])
    else:
        st.warning("⚠️ Advertencia: Columna 'IDRUBRO' no encontrada para ordenar.")
    
    
    # Crear un índice numérico para el eje Y después de ordenar
    tareas_df['y_num'] = range(len(tareas_df))
    
    # --- Crear figura ---
    fig = go.Figure()
    
    fecha_inicio_col = 'FECHAINICIO' in tareas_df.columns 
    fecha_fin_col = 'FECHAFIN' in tareas_df.columns 
    
    if fecha_inicio_col not in tareas_df.columns or fecha_fin_col not in tareas_df.columns:
         st.warning("❌ Error: No se encontraron columnas de fechas de inicio/fin necesarias para dibujar el Gantt.")


    inicio_rubro_calc = tareas_df.set_index('IDRUBRO')[fecha_inicio_col].to_dict()
    fin_rubro_calc = tareas_df.set_index('IDRUBRO')[fecha_fin_col].to_dict()
    # También necesitamos saber si una tarea es crítica para colorear las flechas
    is_critical_dict = tareas_df.set_index('IDRUBRO')['RUTA_CRITICA'].to_dict()
    
    
    # Reconstruir el grafo de dependencias (predecesoras a sucesores)
    # Necesitamos esto para dibujar las flechas correctamente
    dependencias = defaultdict(list)
    # We also need the dependency details for each pre->suc link (type and lag)
    predecesoras_map_details = defaultdict(list) # suc_id -> [(pre_id1, type1, lag1), (pre_id2, type2, lag2), ...]
    
    for _, row in tareas_df.iterrows():
        tarea_id = row['IDRUBRO']
        predecesoras_str = str(row['PREDECESORAS']).strip()
    
        if predecesoras_str not in ['nan', '']:
            pre_list = predecesoras_str.split(',')
            for pre_entry in pre_list:
                pre_entry = pre_entry.strip()
                match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
    
                if match:
                    pre_id = int(match.group(1))
                    tipo_relacion = match.group(2).upper() if match.group(2) else 'FC' # Default to FC
                    desfase = int(match.group(3)) if match.group(3) else 0 # Default to 0 lag
    
                    # Only add dependency if predecessor ID exists in the tasks list
                    if pre_id in tareas_df['IDRUBRO'].values:
                         dependencias[pre_id].append(tarea_id)
                         predecesoras_map_details[tarea_id].append((pre_id, tipo_relacion, desfase))
                    else:
                         st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} mencionado en '{pre_entry}' para tarea {tarea_id} no encontrado en la lista de tareas. Ignorando esta dependencia.")
                else:
                    if pre_entry != '': # Avoid warning for empty strings
                        st.warning(f"⚠️ Advertencia: Formato de predecesora '{pre_entry}' no reconocido para la tarea {tarea_id}. Ignorando.")
    
    
    # --- Agregar franjas horizontales alternadas (basado en y_num) ---
    shapes = []
    # Color gris un poco más oscuro y menos transparente
    color_banda = 'rgba(220, 220, 220, 0.6)'
    
    # Iterar directamente sobre los valores del eje Y (y_num)
    for y_pos in range(len(tareas_df)):
        if y_pos % 2 == 0: # Para posiciones Y pares (0, 2, 4, ...)
            # Añadir un rectángulo que cubra la altura de esta fila
            shapes.append(
                dict(
                    type="rect",
                    xref="paper", # Referencia al área del gráfico (0 a 1)
                    yref="y",    # Referencia al eje Y de datos
                    x0=0,        # Cubre todo el ancho del área del gráfico (en coordenadas del paper)
                    x1=1,        # Cubre todo el ancho del área del gráfico (en coordenadas del paper)
                    y0=y_pos - 0.5, # Cubre desde la mitad inferior de la fila anterior (en coordenadas del eje Y de datos)
                    y1=y_pos + 0.5, # Hasta la mitad superior de la fila siguiente (en coordenadas del eje Y de datos)
                    fillcolor=color_banda,
                    layer="below", # Asegurar que esté detrás de las barras y líneas
                    line_width=0,
                )
            )

    
    color_no_critica_barra = 'lightblue' # Celeste para no críticas
    color_critica_barra = 'rgb(255, 133, 133)' # Rojo específico para críticas
    
    for i, row in tareas_df.iterrows():
        # Determinar color de la barra
        line_color = color_critica_barra if row.get('RUTA_CRITICA', False) else color_no_critica_barra
        line_width = 12 # Ancho estándar para las barras
    
        # Usar las fechas calculadas para dibujar la barra
        start_date = row[fecha_inicio_col]
        end_date = row[fecha_fin_col]
    
        # Asegurarse de que las fechas son válidas antes de dibujar la barra
        if pd.isna(start_date) or pd.isna(end_date):
             st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea {row['RUBRO']} (ID {row['IDRUBRO']}). No se dibujará la barra.")
             continue
    
    
        # --- Formatear costo para el hover (sin locale, formato estándar S/. con separador de miles y 2 decimales) ---
        try:
            valor_costo = float(row.get(cost_column_name, 0))
            costo_formateado = f"S/ {valor_costo:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            # Esto convierte: 12345.67 → S/. 12.345,67
        except Exception:
            costo_formateado = "S/ 0,00"

    
        # Crear texto para hover
        hover_text = (
            f"📌 <b>Rubro:</b> {row['RUBRO']}<br>"
            f"🗓️ <b>Capítulo:</b> {row['CAPÍTULO']}<br>"
            f"📅 <b>Inicio</b> {start_date.strftime('%d/%m/%Y')}<br>"
            f"🏁 <b>Fin:</b> {end_date.strftime('%d/%m/%Y')}<br>"
            f"⏱️ <b>Duración:</b> {(end_date - start_date).days} días<br>"
            f"⏳ <b>Holgura Total:</b> {row.get('HOLGURA_TOTAL', 'N/A')} días<br>" # Mostrar holgura si existe
            f"💰 <b>Costo:</b> {costo_formateado}"
        )
    
        fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[row['y_num'], row['y_num']],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            showlegend=False, # Ocultar leyenda de las barras
            hoverinfo='text',
            text=hover_text, # Usar el texto formateado con HTML
        ))
    
    # --- Flechas de dependencias (ortogonales con 1 o 2 codos) ---
    offset_days_horizontal = 5 # Número de días para el desfase horizontal en los codos (para CC/FC)
    
    # Definir colores específicos para las flechas/líneas de dependencia
    color_no_critica_flecha = 'blue'
    color_critica_flecha = 'red'
    
    # Iterar sobre las dependencias definidas (usando el grafo reconstruido)
    for pre_id, sucesores in dependencias.items():
        # Asegurarse de que la predecesora existe en el DF y tiene un y_num asignado
        pre_row_df = tareas_df[tareas_df['IDRUBRO'] == pre_id]
        if pre_row_df.empty:
            st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} en el grafo de dependencias no encontrado en tareas_df. Saltando flechas desde este ID.")
            continue
        # Usamos .iloc[0] ya que esperamos solo una fila por ID
        y_pre = pre_row_df.iloc[0]['y_num']
        pre_is_critical = is_critical_dict.get(pre_id, False)
    
    
        # Obtener fechas calculadas de la predecesora (usando los diccionarios calculados)
        x_pre_inicio = inicio_rubro_calc.get(pre_id)
        x_pre_fin = fin_rubro_calc.get(pre_id)
    
        # Verificar si las fechas de la predecesora son válidas
        if pd.isna(x_pre_inicio) or pd.isna(x_pre_fin):
             st.warning(f"⚠️ Advertencia: Fechas calculadas no encontradas o inválidas para el predecesor ID {pre_id}. No se dibujarán flechas desde este ID.")
             continue
    
    
        for suc_id in sucesores:
            # Asegurarse de que la sucesora existe en el DF y tiene un y_num asignado
            suc_row_df = tareas_df[tareas_df['IDRUBRO'] == suc_id]
            if suc_row_df.empty:
                st.warning(f"⚠️ Advertencia: Sucesor ID {suc_id} en el grafo de dependencias no encontrado en tareas_df. Saltando flecha hacia este ID.")
                continue
            # Usamos .iloc[0] ya que esperamos solo una fila por ID
            y_suc = suc_row_df.iloc[0]['y_num']
            suc_is_critical = is_critical_dict.get(suc_id, False)
    
            # Determinar color de la flecha: Rojo si AMBOS (predecesor y sucesor) son críticos, Azul en otro caso
            arrow_color = color_critica_flecha if pre_is_critical and suc_is_critical else color_no_critica_flecha
            line_style = dict(color=arrow_color, width=1, dash='dash') # Estilo de línea segmentada
    
            # Obtener fechas calculadas de la sucesora (usando los diccionarios calculados)
            x_suc_inicio = inicio_rubro_calc.get(suc_id)
            x_suc_fin = fin_rubro_calc.get(suc_id)
    
            # Verificar si las fechas de la sucesora son válidas
            if pd.isna(x_suc_inicio) or pd.isna(x_suc_fin):
                 st.warning(f"⚠️ Advertencia: Fechas calculadas no encontradas o inválidas para el sucesor ID {suc_id}. No se dibujará la flecha.")
                 continue
    
            tipo_relacion = 'FC' # Default if not found or parsed correctly
            for pre_id_suc, type_suc, desfase_suc in predecesoras_map_details.get(suc_id, []):
                if pre_id_suc == pre_id: # Found the dependency from pre_id to suc_id
                     tipo_relacion = type_suc.upper() if type_suc else 'FC'
                     # Note: desfase_suc is not needed for determining origin/connection points based on type
                     break # Found the specific dependency details, exit inner loop
    
    
            origin_x = x_pre_fin # Default FC for the graph visualization point
            if tipo_relacion == 'CC':
                origin_x = x_pre_inicio
            elif tipo_relacion == 'CF':
                origin_x = x_pre_inicio
            elif tipo_relacion == 'FF':
                origin_x = x_pre_fin
            # Else handled by default
    
            # --- Determinar el punto de CONEXIÓN en la sucesora (Flecha) ---
            connection_x = x_suc_inicio # Default FC for the graph visualization point
            arrow_symbol = 'triangle-right' # Default FC for the graph visualization point
    
            if tipo_relacion == 'CC':
                connection_x = x_suc_inicio
                arrow_symbol = 'triangle-right'
            elif tipo_relacion == 'CF':
                connection_x = x_suc_fin
                arrow_symbol = 'triangle-left' # User indicated left arrow for CF
            elif tipo_relacion == 'FF':
                connection_x = x_suc_fin
                arrow_symbol = 'triangle-left' # User indicated left arrow for FF
            # Else handled by default
    
    
            # --- Dibujar el círculo en el punto de origen de la predecesora ---
            # Usar el mismo color que la flecha
            fig.add_trace(go.Scattergl(
                x=[origin_x],
                y=[y_pre],
                mode='markers',
                marker=dict(
                    symbol='circle', # Símbolo de círculo
                    size=8, # Tamaño del círculo
                    color=arrow_color, # Color del círculo (mismo que la flecha)
                ),
                hoverinfo='none',
                showlegend=False,
            ))
    
    
            # --- Dibujar los segmentos de línea ortogonal ---
            # Usar el color de la flecha para los segmentos
            line_style = dict(color=arrow_color, width=1, dash='dash') # Estilo de línea segmentada
    
            # Definir los puntos para los segmentos
            points_x = [origin_x]
            points_y = [y_pre]
    
            # Lógica de codos basada en el tipo de relación y el desfase horizontal
            if tipo_relacion in ['CC', 'FC']:
                # CC/FC path: Horizontal Left -> Vertical -> Horizontal Right (con 2 codos)
                # Horizontal a la izquierda desde el origen
                elbow1_x = origin_x - timedelta(days=offset_days_horizontal)
                elbow1_y = y_pre
    
                # Vertical hasta la altura de la sucesora
                elbow2_x = elbow1_x # Misma X que elbow1
                elbow2_y = y_suc   # Misma Y que sucesor
    
                # Segmento 1 (Horizontal Left)
                points_x.append(elbow1_x)
                points_y.append(elbow1_y)
    
                # Segmento 2 (Vertical)
                points_x.append(elbow2_x)
                points_y.append(elbow2_y)
    
                # Segmento 3 (Horizontal Right) hasta el punto de conexión
                points_x.append(connection_x)
                points_y.append(y_suc)
    
    
            elif tipo_relacion in ['CF', 'FF']:
                 # CF/FF path: Vertical -> Horizontal (con 1 codo)
                 # Codo: Misma X que origen, misma Y que sucesor
                 elbow1_x = origin_x # Misma X que origen
                 elbow1_y = y_suc    # Misma Y que sucesor
    
                 # Segmento 1 (Vertical)
                 points_x.append(elbow1_x)
                 points_y.append(elbow1_y)
    
                 # Segmento 2 (Horizontal) hasta el punto de conexión
                 points_x.append(connection_x)
                 points_y.append(y_suc)
    
            else:
                 # Si el tipo no es reconocido, no dibujamos los segmentos (ya se dio advertencia al determinar tipo)
                 st.warning(f"⚠️ Tipo de relación '{tipo_relacion}' no reconocido para dibujar segmentos de flecha entre ID {pre_id} y ID {suc_id}. Saltando flecha.")
                 continue
    
    
            # Dibujar los segmentos usando un solo Scatter con mode='lines'
            fig.add_trace(go.Scatter(
                x=points_x,
                y=points_y,
                mode='lines',
                line=line_style,
                hoverinfo='none', # No mostrar hover para la línea
                showlegend=False,
            ))
    
    
            # --- Añadir flecha como marcador en el punto de conexión de la sucesora ---
            # Usar el mismo color que la flecha
            fig.add_trace(go.Scattergl(
                x=[connection_x],
                y=[y_suc],
                mode='markers',
                marker=dict(
                    symbol=arrow_symbol, # Símbolo de flecha (izquierda o derecha)
                    size=10, # Tamaño de la flecha
                    color=arrow_color, # Color de la flecha (mismo que la línea)
                ),
                hoverinfo='none',
                showlegend=False,
            ))
    
    
    # --- Preparar etiquetas del eje Y con estilo alternado (Negrita) ---
    y_ticktext_styled = []
    # Iterar directamente sobre los valores de y_num para alinear estilo con posición visual
    for y_pos in range(len(tareas_df)):
        # Encontrar la fila del DataFrame que corresponde a esta posición y_num
        row_for_y_pos = tareas_df[tareas_df['y_num'] == y_pos]
        if not row_for_y_pos.empty:
            rubro_text = row_for_y_pos.iloc[0]['RUBRO']
            # Estilo alternado: negrita cada dos filas
            y_ticktext_styled.append(f"<b>{rubro_text}</b>" if y_pos % 2 == 0 else rubro_text)
    
        else:
             # Esto no debería pasar si y_num está bien asignado, pero como fallback
             y_ticktext_styled.append("") # Añadir una cadena vacía si no se encuentra el rubro
    
    
    fig.update_layout(
        xaxis=dict(
            title='Fechas',
            # tickformat=date_tick_format, # Eliminar formato explícito para usar default
            side='bottom', # Asegurar que el eje principal está abajo
            dtick='M1', # Forzar ticks principales mensualmente
            tickangle=-90, # Rotar etiquetas a -90 grados (vertical) para eje inferior
            showgrid=True, # Mostrar cuadrícula vertical
            gridcolor='rgba(128,128,128,0.3)', # Color gris semi-transparente para la cuadrícula
            gridwidth=0.5
        ),
        xaxis2=dict(
            title='Fechas',
            # tickformat=date_tick_format, # Eliminar formato explícito para usar default
            overlaying='x', # Superponer sobre el eje X principal
            side='top', # Posicionar arriba
            dtick='M1', # Forzar ticks principales mensualmente
            tickangle=90, # Rotar etiquetas a 90 grados (vertical) para eje superior
            showgrid=True, # Mostrar cuadrícula vertical
            gridcolor='rgba(128,128,128,0.3)', # Color gris semi-transparente para  cuadrícula
            gridwidth=0.5
        ),
        yaxis_title='Rubro',
        yaxis=dict(
            autorange='reversed',
            tickvals=tareas_df['y_num'],
            ticktext=y_ticktext_styled, # Usar la lista de etiquetas con estilo
            tickfont=dict(size=10),
            showgrid=False # No mostrar cuadrícula horizontal por defecto (las bandas la reemplazan)
        ),
        shapes=shapes, # Añadimos las formas de las franjas horizontales
        height=max(600, len(tareas_df) * 25), # Ajustar altura según número de tareas
        showlegend=False, # Ocultar leyenda de las barras
        plot_bgcolor='white',
        hovermode='closest'
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
    # Recursos
    # Ensure date columns in tareas_df are datetime
    tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'])
    tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'])
    
    # Clean 'RUBRO' and 'CAN' columns for merging
    tareas_df['RUBRO'] = tareas_df['RUBRO'].str.strip()
    dependencias_df['RUBRO'] = dependencias_df['RUBRO'].str.strip()
        
    recursos_tareas_df = dependencias_df.merge(
        tareas_df[['IDRUBRO', 'RUBRO', 'FECHAINICIO', 'FECHAFIN', 'DURACION']],
        left_on='RUBRO',
        right_on='RUBRO',
        how='left'
    )
    
    # List to hold daily resource usage dataframes for each task/resource combination
    daily_resource_usage_list = []
    
    # Iterate over each row in the merged DataFrame
    for index, row in recursos_tareas_df.iterrows():
        task_id = row['IDRUBRO']
        resource_name = row['RECURSO']
        unit = row['UNIDAD']
        total_quantity = row['CANTIDAD']
        start_date = row['FECHAINICIO']
        end_date = row['FECHAFIN']
        duration_days = row['DURACION']
    
        # Ensure dates are valid and duration is non-negative
        if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
            st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea ID {task_id}, recurso '{resource_name}'. Saltando.")
            continue
    
        if duration_days <= 0:
            daily_quantity = total_quantity
            date_range = [start_date]
        else:
            # Calculate daily quantity assuming uniform distribution
            daily_quantity = total_quantity / (duration_days + 1) # Divide by number of days including start and end
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        temp_df = pd.DataFrame({
            'Fecha': date_range,
            'IDRUBRO': task_id,
            'RECURSO': resource_name,
            'UNIDAD': unit,
            'Cantidad_Diaria': daily_quantity,
            'Cantidad_Total_Tarea': total_quantity # Keep total quantity for reference
        })
    
        # Append the temporary DataFrame to the list
        daily_resource_usage_list.append(temp_df)

    # Concatenate all temporary DataFrames into one
    if daily_resource_usage_list:
        all_daily_resource_usage_df = pd.concat(daily_resource_usage_list, ignore_index=True)
    else:
        st.warning("\nNo se generaron datos de uso diario de recursos.")
        all_daily_resource_usage_df = pd.DataFrame() # Create an empty DataFrame

    # Group by Date, Resource, and Unit to sum up daily quantities
    daily_resource_demand_df = all_daily_resource_usage_df.groupby(
        ['Fecha', 'RECURSO', 'UNIDAD'],
        as_index=False
    )['Cantidad_Diaria'].sum()
    
    # Rename the aggregated quantity column for clarity
    daily_resource_demand_df.rename(columns={'Cantidad_Diaria': 'Demanda_Diaria_Total'}, inplace=True)
    
    daily_resource_demand_df['RECURSO'] = daily_resource_demand_df['RECURSO'].str.strip()
    recursos_df['RECURSO'] = recursos_df['RECURSO'].str.strip()
    
    resource_demand_with_details_df = daily_resource_demand_df.merge(
        recursos_df[['RECURSO', 'TYPE', 'TARIFA']],
        on='RECURSO',
        how='left'
    )

    resource_demand_with_details_df['Costo_Diario'] = resource_demand_with_details_df['Demanda_Diaria_Total'] * resource_demand_with_details_df['TARIFA']
    
    # Group by Date and Type to sum daily costs
    daily_cost_by_type_df = resource_demand_with_details_df.groupby(
        ['Fecha', 'TYPE'],
        as_index=False
    )['Costo_Diario'].sum()
    
    # Group by Date and Resource to sum daily quantities
    daily_demand_by_resource_df = resource_demand_with_details_df.groupby(
        ['Fecha', 'RECURSO', 'UNIDAD'],
        as_index=False
    )['Demanda_Diaria_Total'].sum()

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
            line=dict(color=pastel_blue, width=10), # Use pastel blue for all bars
            name=row['RECURSO'], # Name for potential legend (though we will hide it)
            showlegend=False, # Hide default legend
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
            tickfont=dict(size=10) # Adjust font size if needed
        ),
        xaxis=dict(
            title='Fechas',
            side='bottom',
            dtick='M1',  # Monthly ticks
            tickangle=-90, # Vertical labels
            showgrid=True,
            gridcolor='rgba(128,128,128,0.3)',
            gridwidth=0.5
        ),
        # Add a top x-axis for symmetry/clarity if desired (optional, mirroring bottom axis)
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
        height=max(600, len(recursos_tareas_df['RECURSO'].unique()) * 20), # Adjust height based on number of unique resources
        showlegend=False, # Hide default legend
        plot_bgcolor='white',
        hovermode='closest'
    )

    st.plotly_chart(fig_resource_timeline, use_container_width=True)


else:
    st.warning("Sube el archivo Excel con las hojas Tareas, Recursos y Dependencias.")























