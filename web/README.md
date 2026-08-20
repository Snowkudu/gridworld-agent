# GridWorld Lab web client

The P6 client is a static Vite application. ONNX Runtime Web executes the model
in the visitor's browser; no Python server is involved.

## Model assets

Export the P5 checkpoint directly into the public model directory:

```powershell
python -m scripts.export_onnx `
  --checkpoint "artifacts\p5_dqn\p5_story_frozen_transfer\checkpoint.pt" `
  --output "web\public\models\frozen_transfer.onnx"
```

The exporter creates both `frozen_transfer.onnx` and
`frozen_transfer.json`. Vite copies files in `public/` into the static build
without transforming them.

## Local development

```powershell
cd web
npm install
npm run dev
```

Run the focused JavaScript tests and production build with:

```powershell
npm test
npm run build
```
