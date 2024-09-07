"""

    Sample to show Gemini with caching to execute agentic reflection
    User upload PDF - gemini create summarization

"""
import os
import asyncio
import datetime

from uuid import uuid4

import flet as ft

from vertexai.generative_models import GenerativeModel, Content, Part

MODEL_NAME = 'gemini-1.5-pro-001'
ROOT_DIR = '/tmp/flet'


async def main(page: ft.Page):

    async def get_file(e: ft.FilePickerResultEvent):

        async def update_text(text: str):
            progress_text.value = text
            page.update()

        if e.files:
            progress_ring.visible = True
            progress_text.visible = True
            await update_text('Downloading')

            file = e.files[0]

            tmp_dir_name = str(uuid4())[:4]
            upload_file = ft.FilePickerUploadFile(
                file.name,
                upload_url=e.page.get_upload_url(f'{tmp_dir_name}/{file.name}', 60)
            )
            file_name = f"{ROOT_DIR}/{tmp_dir_name}/{file.name}"

            pick_file_dialog.upload([upload_file])

            downloaded = False
            while not downloaded:
                try:
                    with open(file_name, "rb") as f:
                        data = f.read()
                    downloaded = True
                except FileNotFoundError:
                    await asyncio.sleep(1)

            contents = [Part.from_data(data, "application/pdf")]

            await update_text('Running')

            summaries = []
            writer = [Content(role='USER', parts=[
                Part.from_text('<PAPER>'), 
                Part.from_data(data, 'application/pdf'), 
                Part.from_text('</PAPER>')])]
            reviewer = []

            writer_model = GenerativeModel(model_name='gemini-1.5-flash-001', system_instruction='Create summary of the paper. When you get the feedback respond with the updated summary.')
            review_model = GenerativeModel(model_name='gemini-1.5-flash-001', system_instruction='Review user submitted summary of the paper. Provide concise, actionable feedback which can user utilize to improve the summary')

            for i in range(3):
                await update_text(f'Generating round {i+1}')
                writer_response = writer_model.generate_content(writer)
                if i == 0:
                    reviewer_parts = [Part.from_data(data, 'application/pdf')]
                else:
                    reviewer_parts = []

                reviewer_parts.append(writer_response.candidates[0].content.parts[0])
                reviewer.append(Content(role='USER', parts=reviewer_parts))
                writer.append(Content(role='MODEL', parts=[writer_response.candidates[0].content.parts[0]]))
                summaries.append(writer_response.candidates[0].content.parts[0].text)

                await update_text(f'Reviewing round {i+1}')
                reviewer_response = review_model.generate_content(reviewer)
                writer.append(Content(role='USER', parts=[reviewer_response.candidates[0].content.parts[0]]))
                reviewer.append(Content(role='MODEL', parts=[reviewer_response.candidates[0].content.parts[0]]))
                
            tabs = []
            for idx, summary in enumerate(summaries):
                tabs.append(
                    ft.Tab(
                        text = f'Iteration {idx+1}',
                        content = ft.Container(
                            content=ft.Text(summary)
                        )
                    )
                )
            outputs.controls = [
                ft.Tabs(
                    selected_index=len(summaries)+1,
                    tabs=tabs
                )
            ]

            await update_text('Done')


        else:
            outputs.controls[0].value = "Cancelled!"

         

    page.title = "Gemini Reflection sample"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    # example for generating page theme colors based on the seed color
    page.theme = ft.Theme(color_scheme_seed="blue")

    pick_file_dialog = ft.FilePicker(on_result=get_file)
    pick_file_button = ft.ElevatedButton(
                    text="Select PDF File",
                    icon=ft.icons.UPLOAD_FILE,
                    # on_click=run
                    on_click=lambda _: pick_file_dialog.pick_files(
                        allow_multiple=False
                    ),
                )

    progress_ring = ft.ProgressRing(width=20, height=20, visible=False)
    progress_text = ft.Text("Processing", visible=False)

    file_column = ft.Column(
            alignment=ft.alignment.top_left,
            controls=[
                pick_file_button,
                ft.Row(controls = [progress_ring, progress_text])

            ]
        )

    file_container = ft.Container(
        height=page.height,
        width=200,
        content=file_column
    )

    page.overlay.append(pick_file_dialog)

    outputs = ft.Column(col=9, controls=[ft.Text("Response will be here")])

    page.add(
        ft.AppBar(
            title=ft.Text("Gemini Reflection samples"),
            center_title=True,
            # TODO: reset button
#             actions=[pick_file_button]
        ),
        ft.ResponsiveRow(
            [
                ft.Column(
                    col=2,
                    controls=[file_container]),
                ft.Column(col=1, controls=[ft.VerticalDivider()]),
                outputs
            ]
        )
    )

os.environ["FLET_SECRET_KEY"] = "SECRET_KEY"
ft.app(main, upload_dir=ROOT_DIR, view=ft.AppView.WEB_BROWSER)
