---
name: finish-phase
description: Procedimiento obligatorio al terminar cada fase del proyecto fraud_detector. Usar cuando una tarea de tasks.md/fase esté completa y haya que cerrar la fase (commit e push a GitHub). Disparar con frases como "termina la fase", "cerra la fase", "commit y push", "cierre de fase".
---

# Cierre de fase (finish-phase)

Cada vez que se termina una fase de `tasks.md`, ejecutar este procedimiento **completo y en orden** antes de declararla cerrada. El repo remoto es `github.com/matiasbelsito7/fraud_detector.git` (origin).

## 1. Estado del trabajo

- Verificar que todos los pasos/entregables de la fase están completos según `tasks.md`.
- Asegurarse de que los datos y artefactos grandes no estén trackeados (`.gitignore`).
- No dejar archivos secretos trackeables (`api_*.txt`, tokens).

## 2. Chequeos de calidad (en este orden, todos contra `.venv`)

Usar el intérprete del entorno virtual. Ante cualquier fallo, corregir el código y repetir el check hasta que pase.

1. **ruff** (lint + fix automático):
   ```
   .\.venv\Scripts\python.exe -m ruff check src run_*.py
   .\.venv\Scripts\python.exe -m ruff check --fix src run_*.py
   ```
   Nota: en PowerShell el glob `run_*.py` NO se expande solo; listar los scripts existentes explícitamente (p. ej. `run_data_validation.py`).
2. **black** (formato):
   ```
   .\.venv\Scripts\python.exe -m black --check src run_data_validation.py
   ```
   Si falla, formatear: `.\.venv\Scripts\python.exe -m black src run_data_validation.py`
3. **mypy** (tipado):
   ```
   .\.venv\Scripts\mypy.exe src run_data_validation.py
   ```

## 3. Pre-commit

- Instalar los hooks una única vez si aún no están: `.\.venv\Scripts\pre-commit.exe install`
- Ejecutar sobre todo el repo:
  ```
  .\.venv\Scripts\pre-commit.exe run --all-files
  ```
- Si un hook modifica/falla, arreglar lo señalado y volver a correr hasta que pase.

## 4. Commit

- Revisar `git status` y `git diff`; stagear solo lo intencional.
- Mensaje con formato convencional y referencia a la fase, p. ej.:
  - `docs: definir agents, constitution y specs` (proyecto)
  - `build(data): validar datos IEEE-CIS y validar estructura` (fase B)
  - `feat(eda): análisis exploratorio ...` (fase C)
- No commitear datos (`data/`), `reports/` (si están ignorados), ni secretos.

## 5. Push

- Verificar que el remote `origin` apunte a `https://github.com/matiasbelsito7/fraud_detector.git`:
  ```
  git remote -v
  ```
  Si no existe: `git remote add origin https://github.com/matiasbelsito7/fraud_detector.git`
- Hacer push de la rama actual:
  ```
  git push -u origin <rama_actual>
  ```
  (rama actual habitual: `main`)

## 6. Registro

- Actualizar `README.md` ("Estado actual") y, si corresponde, `tasks.md` reflejando la fase completada.
- Reportar al usuario: checks pasados, hash del commit y confirmación del push.