Power BI Dashboard build instructions

Overview

This folder contains the sample data and the Learner Guide PDF used to create a Power BI report that incorporates the Learner Guide. Because Power BI Desktop is not available here, follow the steps below on your local machine (Power BI Desktop) to build the .pbix and export a .pbit template.

Files in this folder (relative to repository root):
- AI_LSI_Demo_Historical_Data - Copy.csv  (time series sample data)
- Learner Guide.pdf                         (the Learner Guide to embed)
- report_spec.json                          (visuals & DAX spec)

Pre-requisites

- Power BI Desktop (latest stable) installed on your Windows machine
- (Optional but recommended) PowerPoint, Adobe Acrobat, or an image-export tool to convert PDF pages to PNG images for embedding
- Internet access to download any Power BI custom visuals if desired

High-level plan

1. Import the CSV into Power BI Desktop and clean types
2. Build the data model (single table) and create measures
3. Create report pages and visuals following the spec
4. Embed the Learner Guide (either via PDF Viewer custom visual using a hosted URL, or convert PDF pages to images and add them as pages)
5. Export the report as a Power BI Template (.pbit) and keep the assets next to it

Detailed steps

1) Open Power BI Desktop

2) Get Data
- Home > Get Data > Text/CSV
- Browse to the CSV in the repository path:
  C:\Users\UM_AS\OneDrive\Documents.worktrees\pbix-dashboard-learner-guide-integration\AI_LSI_Demo_Historical_Data - Copy.csv
- Click Load (or Transform Data to open Power Query for light cleaning)

3) Power Query (recommended transforms)
- Ensure Date column is detected as Date/Time. If not: select Date column > Data Type > Date/Time
- Rename the table to "LSI_Historical" (or keep default)
- Check numeric columns (Temperature_C, Flow_m3_h, pH, Calcium_mg_L, Alkalinity_mg_L, TDS_mg_L, LSI) are numeric
- If you want a Date-only column, add a transform: Add Column > Date > Date Only from DateTime
- Apply & Close

4) Data model
- This dataset is a single table time-series; no joins required

5) Create recommended measures (DAX)
- In the Fields pane, right-click the table > New measure and paste the DAX formulas below (replace table name if different):

-- Latest value for a numeric column (example: Temperature)
Latest Temperature =
VAR LastDate = MAX('LSI_Historical'[Date])
RETURN
CALCULATE(AVERAGE('LSI_Historical'[Temperature_C]), FILTER('LSI_Historical','LSI_Historical'[Date] = LastDate))

-- Average over visible filter context
Avg Temperature = AVERAGE('LSI_Historical'[Temperature_C])

-- 7-day moving average of LSI (assumes daily data)
LSI 7d MA =
CALCULATE(
    AVERAGE('LSI_Historical'[LSI]),
    DATESINPERIOD('LSI_Historical'[Date], LASTDATE('LSI_Historical'[Date]), -7, DAY)
)

-- Max Flow
Max Flow = MAX('LSI_Historical'[Flow_m3_h])

-- Latest LSI value
Latest LSI =
VAR LastDate = MAX('LSI_Historical'[Date])
RETURN
CALCULATE(LASTNONBLANKVALUE('LSI_Historical'[Date], FIRSTNONBLANK('LSI_Historical'[LSI],1)), FILTER('LSI_Historical','LSI_Historical'[Date]=LastDate))

Note: adjust measure logic as needed; some visuals like Cards/Tiles work best with simple measures (SUM, AVERAGE, MAX)

6) Create the report pages and visuals (suggested layout)

Page 1 - Overview / Executive
- Title: "LSI Monitoring — Overview"
- Top row: KPI cards for Latest Temperature, Latest LSI, Max Flow (create Card visuals and use the measures)
- Middle: Multi-row card or table showing latest values for key chemistry columns (pH, Calcium, Alkalinity, TDS)
- Bottom: Line chart (Time series)
  - Axis: Date
  - Values: Temperature_C, Flow_m3_h (use separate Y-axes if needed) or create separate charts stacked vertically
- Slicer: Date range (use Date field with between slider)

Page 2 - LSI Analysis
- Line chart: LSI over time
  - Axis: Date
  - Values: LSI and add the "LSI 7d MA" measure as a separate line
- Visual: Scatter or correlation chart of LSI vs Temperature or LSI vs TDS

Page 3 - Chemistry Details
- Table or Matrix showing Date, pH, Calcium_mg_L, Alkalinity_mg_L, TDS_mg_L
- Conditional formatting: highlight pH or LSI outside target thresholds

Page 4 - Learner Guide (embedded)
Option A — Use PDF Viewer custom visual (if you can host the PDF externally):
  1. In Power BI Desktop, open the Visualizations pane > Get more visuals (three dots) > From AppSource
  2. Search for "PDF Viewer" and add it
  3. Add the PDF Viewer visual to the page and set the URL property to the hosted URL of Learner Guide.pdf (e.g., upload the PDF to SharePoint/OneDrive and use the sharing link). The visual will render the PDF inside the report.

Option B — Convert the PDF to images and add each page as a report page background or image visual (works offline):
  1. Open Learner Guide.pdf in PowerPoint (Insert > PDF) or Adobe Acrobat and export pages to PNGs
  2. Save the images into the repo folder (e.g. powerbi-assets\learner-pages\page1.png, page2.png...)
  3. In Power BI Desktop, create a new page for each exported image. Insert > Image, then select the corresponding PNG. Add navigation buttons (Bookmarks + Buttons) if you want to provide Prev/Next.

Option C — Reference the PDF via a link in a Text box
  1. Insert a text box on a page and paste the file path or a hosted link to the Learner Guide. Clicking the link will open it externally.

7) Formatting and interactions
- Use Sync slicers across pages for consistent date selection (View > Sync slicers)
- Add tooltips to charts explaining how to interpret LSI
- Set default page size to 16:9 or Letter as preferred
- Use bookmarks to create a guided flow (Overview -> Analysis -> Learner Guide)

8) Test and publish
- Verify visuals update with the Date slicer
- File > Save as report (*.pbix)
- File > Export > Power BI template (*.pbit) to produce a reusable template (it will prompt to include sample data or not)

9) Packaging assets
- Keep the CSV and Learner Guide.pdf with the .pbit in the same folder for reproducibility
- If you used external images for the Learner Guide, include them too

Troubleshooting / Notes

- Embedding a local PDF directly inside Power BI Desktop is limited; the PDF Viewer visual generally expects a web URL. For offline embedding, use exported images.
- If you want the Learner Guide to be searchable inside Power BI, consider extracting text into a table (one record per page or section) and adding a slicer/filterable text box. That is more advanced and requires splitting and pasting the text into a CSV or Excel file.

If you want, the next actions can be:
- Convert PDF pages to PNGs here and add them to powerbi-assets so you can insert images directly (Confirm and I will export pages to PNGs locally if tools available).
- Create the DAX measures and report_spec.json (already included) and tweak the visuals list.

