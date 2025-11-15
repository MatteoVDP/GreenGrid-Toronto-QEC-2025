# Toronto GeoJSON Interactive Map

An interactive web application for visualizing Toronto GeoJSON data with zoom and pan capabilities.

## Features

- Interactive map view using Leaflet.js
- Zoom in/out with mouse wheel or controls
- Pan by dragging the map
- Color-coded areas based on ML Priority Score
- Hover to view area information
- Click on areas to zoom in
- Responsive design

## Usage

1. Open `index.html` in a web browser
   - Simply double-click the file, or
   - Use a local web server (recommended for best performance)

2. Using a local web server (recommended):
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Using Python 2
   python -m SimpleHTTPServer 8000
   
   # Using Node.js (with http-server installed)
   npx http-server
   ```
   
   Then open `http://localhost:8000` in your browser.

## Map Controls

- **Zoom In/Out**: Use mouse wheel, or click the +/- buttons, or use pinch gesture on touch devices
- **Pan**: Click and drag the map
- **Area Information**: Hover over any area to see details in the info panel
- **Zoom to Area**: Click on any area to zoom into it

## Data

The map displays data from `toronto_with_predictions.geojson` with the following properties:
- DAUID (Dissemination Area Unique Identifier)
- ML Priority Score
- Heat Normalized
- Tree Normalized
- Vulnerability Normalized
- Footfall Normalized

## Color Legend

The map uses color coding based on ML Priority Score:
- Dark Green: Low (0.0 - 0.3)
- Green: Low-Medium (0.3 - 0.5)
- Light Green: Medium (0.5 - 0.6)
- Yellow-Green: Medium-High (0.6 - 0.7)
- Light Yellow: High (0.7 - 0.8)
- Tan: Very High (0.8+)

