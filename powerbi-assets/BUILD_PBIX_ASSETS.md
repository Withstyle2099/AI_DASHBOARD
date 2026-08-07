Package contents in this folder (created by the assistant):
- dax_measures.txt            -- DAX measures to paste into Power BI
- powerquery_m.txt           -- Power Query M script to import & clean the CSV
- powerbi-theme.json         -- Simple Power BI theme JSON to import
- POWERBI_BUILD_INSTRUCTIONS.md (existing) -- high-level steps (keep using this)

Goal: produce these artifacts so you can open Power BI Desktop locally and finish the .pbix file.

1) PDF -> PNG (local steps)

Recommended (ImageMagick):
  magick -density 150 "C:\\path\\to\\powerbi-assets\\Learner Guide.pdf" "C:\\path\\to\\powerbi-assets\\learner-pages\\page-%03d.png"

Poppler (pdftoppm):
  pdftoppm -png -r 150 "C:\\path\\to\\powerbi-assets\\Learner Guide.pdf" "C:\\path\\to\\powerbi-assets\\learner-pages\\page"

PowerPoint / Adobe Acrobat: open the PDF and export each page as PNG images. Save them into powerbi-assets\\learner-pages.

2) Data import (Power BI Desktop)
- Open Power BI Desktop
- Get Data > Blank Query -> Advanced Editor
- Paste contents of powerquery_m.txt (adjust File.Contents path if needed) and click Done
- Rename the query/table to LSI_Historical and Apply & Close

3) Measures and visuals
- Open dax_measures.txt and create each New measure in the LSI_Historical table
- Import powerbi-theme.json: View -> Themes -> Browse for themes -> select the JSON
- Create pages per report_spec.json or POWERBI_BUILD_INSTRUCTIONS.md (Overview, LSI Analysis, Chemistry Details, Learner Guide)

4) Embedding Learner Guide pages
Option A (offline, recommended here): export PDF pages to PNG and insert each page as an Image visual on a report page. Use Buttons + Bookmarks if you want next/prev navigation.
Option B (online): host Learner Guide.pdf on OneDrive/SharePoint and use PDF Viewer custom visual (AppSource) with the file URL.

5) Finalize
- Save the report as .pbix
- File -> Export -> Power BI template (.pbit) to create a reusable template

Notes & troubleshooting
- If you exported PNGs, create a folder powerbi-assets\\learner-pages and place page-001.png, page-002.png, ... there
- If Date fields import as text, use Power Query to change types before applying
- If you want the assistant to attempt conversion again, confirm which installer to allow (winget or Chocolatey); installation may be blocked in this environment.

If you want, next I can:
- Update POWERBI_BUILD_INSTRUCTIONS.md to include exact paste-ready snippets (done: created scripts)
- Create a small README in powerbi-assets describing the package (confirm)