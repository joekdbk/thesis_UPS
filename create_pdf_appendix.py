#!/usr/bin/env python3
"""
PDF Appendix Generator for UPS Thesis Analysis

This script creates a comprehensive PDF appendix containing voltage plots, current plots,
THD plots, and THD tables for all test events. The PDF is formatted for use as a thesis appendix
with professional typography and consistent layout.

Author: Thesis Analysis System
Date: July 29, 2025
"""

import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re
from datetime import datetime
import glob

class AppendixGenerator:
    def __init__(self, output_folder="output_plots", output_filename="UPS_Analysis_Appendix.pdf"):
        self.output_folder = output_folder
        self.output_filename = output_filename
        self.page_width = A4[0]
        self.page_height = A4[1]
        self.margin = 1*inch
        self.content_width = self.page_width - 2*self.margin
        self.content_height = self.page_height - 2*self.margin
        
        # Try to register Times New Roman font (may not be available on all systems)
        try:
            # Try to register Times New Roman if available
            times_font_paths = [
                "C:/Windows/Fonts/times.ttf",
                "C:/Windows/Fonts/Times.ttf", 
                "/System/Library/Fonts/Times.ttc",
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
            ]
            
            for font_path in times_font_paths:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('TimesRoman', font_path))
                    break
            else:
                print("Times New Roman font not found, using default font")
        except Exception as e:
            print(f"Could not register Times New Roman font: {e}")
            print("Using default font instead")
        
        self.setup_styles()
        self.events = self.discover_events()
        
    def setup_styles(self):
        """Set up document styles with Times New Roman font."""
        self.styles = getSampleStyleSheet()
        
        # Try to use Times New Roman, fall back to Times-Roman if not available
        base_font = 'TimesRoman' if 'TimesRoman' in pdfmetrics.getRegisteredFontNames() else 'Times-Roman'
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='AppendixTitle',
            parent=self.styles['Title'],
            fontName=base_font,
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.black
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading1'],
            fontName=base_font,
            fontSize=14,
            spaceBefore=20,
            spaceAfter=12,
            alignment=TA_LEFT,
            textColor=colors.black
        ))
        
        # Subsection heading style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading2'],
            fontName=base_font,
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            alignment=TA_LEFT,
            textColor=colors.black
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='AppendixBody',
            parent=self.styles['Normal'],
            fontName=base_font,
            fontSize=10,
            spaceBefore=6,
            spaceAfter=6,
            alignment=TA_LEFT,
            textColor=colors.black
        ))
        
        # Table caption style
        self.styles.add(ParagraphStyle(
            name='TableCaption',
            parent=self.styles['Normal'],
            fontName=base_font,
            fontSize=9,
            spaceBefore=8,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=colors.black
        ))
        
        # Figure caption style
        self.styles.add(ParagraphStyle(
            name='FigureCaption',
            parent=self.styles['Normal'],
            fontName=base_font,
            fontSize=9,
            spaceBefore=8,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=colors.black
        ))

    def discover_events(self):
        """Discover all unique test events from the output folder."""
        events = {}
        
        # Pattern to extract event information from filenames
        pattern = r'(voltage|current|THD_plot|THD_table)_INV(-?\d+)_RLC(\d+)_T(\d+)_E(\d+)_.*\.(png|csv)'
        
        for filename in os.listdir(self.output_folder):
            match = re.match(pattern, filename)
            if match:
                file_type, inv_power, rlc_power, trial, event_num, extension = match.groups()
                
                event_key = (int(event_num), int(trial), int(inv_power), int(rlc_power))
                
                if event_key not in events:
                    events[event_key] = {
                        'event_num': int(event_num),
                        'trial': int(trial),
                        'inv_power': int(inv_power),
                        'rlc_power': int(rlc_power),
                        'voltage_plot': None,
                        'current_plot': None,
                        'thd_plot': None,
                        'thd_table': None
                    }
                
                full_path = os.path.join(self.output_folder, filename)
                
                if file_type == 'voltage' and extension == 'png':
                    events[event_key]['voltage_plot'] = full_path
                elif file_type == 'current' and extension == 'png':
                    events[event_key]['current_plot'] = full_path
                elif file_type == 'THD_plot' and extension == 'png':
                    events[event_key]['thd_plot'] = full_path
                elif file_type == 'THD_table' and extension == 'csv':
                    events[event_key]['thd_table'] = full_path
        
        # Sort events by event number, then by trial
        sorted_events = sorted(events.values(), key=lambda x: (x['event_num'], x['trial']))
        
        print(f"Discovered {len(sorted_events)} unique test events")
        return sorted_events

    def format_power_string(self, power):
        """Format power value for display."""
        if power >= 0:
            return f"{power}W"
        else:
            return f"{power}W"

    def create_thd_table_content(self, csv_path):
        """Read THD table CSV and format it for PDF inclusion."""
        try:
            df = pd.read_csv(csv_path, index_col=0)
            
            # Round values to 2 decimal places for readability
            df = df.round(2)
            
            # Create table data
            table_data = []
            
            # Header row
            header = ['Cycle'] + list(df.columns)
            table_data.append(header)
            
            # Data rows
            for idx, row in df.iterrows():
                row_data = [str(idx)] + [f"{val:.2f}" if pd.notna(val) else "N/A" for val in row]
                table_data.append(row_data)
            
            return table_data
            
        except Exception as e:
            print(f"Error reading THD table {csv_path}: {e}")
            return [["Error", "Could not read THD table"]]

    def create_pdf_content(self):
        """Create the complete PDF content."""
        story = []
        
        # Title page
        story.append(Paragraph("Appendix: UPS System Analysis Results", self.styles['AppendixTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Introduction
        intro_text = """
        This appendix contains comprehensive analysis results for the UPS system fault testing. 
        Each test event includes voltage waveforms, current waveforms, Total Harmonic Distortion (THD) plots, 
        and detailed THD data tables. The tests cover various inverter power settings and load conditions 
        to characterize the system's response to electrical faults.
        """
        story.append(Paragraph(intro_text, self.styles['AppendixBody']))
        story.append(Spacer(1, 0.3*inch))
        
        # Test parameters explanation
        params_text = """
        <b>Test Parameters:</b><br/>
        • <b>INV Power:</b> Inverter power setting (negative values indicate power flowing to grid)<br/>
        • <b>RLC Power:</b> Resistive-Inductive-Capacitive load power<br/>
        • <b>Trial:</b> Test repetition number for statistical validation<br/>
        • <b>Event:</b> Unique event identifier for each fault scenario
        """
        story.append(Paragraph(params_text, self.styles['AppendixBody']))
        story.append(PageBreak())
        
        # Process each event
        for i, event in enumerate(self.events):
            event_num = event['event_num']
            trial = event['trial']
            inv_power = event['inv_power']
            rlc_power = event['rlc_power']
            
            # Section header for each event
            section_title = f"Event {event_num} - Trial {trial}"
            story.append(Paragraph(section_title, self.styles['SectionHeading']))
            
            # Event parameters
            params = f"Inverter Power: {self.format_power_string(inv_power)}, Load Power: {self.format_power_string(rlc_power)}"
            story.append(Paragraph(params, self.styles['AppendixBody']))
            story.append(Spacer(1, 0.2*inch))
            
            # Voltage waveforms
            if event['voltage_plot'] and os.path.exists(event['voltage_plot']):
                story.append(Paragraph("Voltage Waveforms", self.styles['SubsectionHeading']))
                
                # Calculate image size to fit page width
                img_width = self.content_width * 0.9
                img_height = img_width * 0.6  # Maintain aspect ratio
                
                voltage_img = Image(event['voltage_plot'], width=img_width, height=img_height)
                story.append(voltage_img)
                
                caption = f"Figure {i*4+1}: Voltage waveforms for Event {event_num}, Trial {trial}"
                story.append(Paragraph(caption, self.styles['FigureCaption']))
                story.append(Spacer(1, 0.2*inch))
            
            # Current waveforms
            if event['current_plot'] and os.path.exists(event['current_plot']):
                story.append(Paragraph("Current Waveforms", self.styles['SubsectionHeading']))
                
                current_img = Image(event['current_plot'], width=img_width, height=img_height)
                story.append(current_img)
                
                caption = f"Figure {i*4+2}: Current waveforms for Event {event_num}, Trial {trial}"
                story.append(Paragraph(caption, self.styles['FigureCaption']))
                story.append(Spacer(1, 0.2*inch))
            
            # THD plots
            if event['thd_plot'] and os.path.exists(event['thd_plot']):
                story.append(Paragraph("Total Harmonic Distortion Analysis", self.styles['SubsectionHeading']))
                
                thd_img = Image(event['thd_plot'], width=img_width, height=img_height)
                story.append(thd_img)
                
                caption = f"Figure {i*4+3}: THD analysis for Event {event_num}, Trial {trial}"
                story.append(Paragraph(caption, self.styles['FigureCaption']))
                story.append(Spacer(1, 0.2*inch))
            
            # THD data table
            if event['thd_table'] and os.path.exists(event['thd_table']):
                story.append(Paragraph("THD Data Table", self.styles['SubsectionHeading']))
                
                table_data = self.create_thd_table_content(event['thd_table'])
                
                # Create table with formatting
                table = Table(table_data, repeatRows=1)
                
                # Table styling
                table.setStyle(TableStyle([
                    # Header styling
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    
                    # Data styling
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    
                    # Alternating row colors
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]))
                
                story.append(table)
                
                caption = f"Table {i+1}: THD values (%) for Event {event_num}, Trial {trial}"
                story.append(Paragraph(caption, self.styles['TableCaption']))
            
            # Add page break between events (except for the last one)
            if i < len(self.events) - 1:
                story.append(PageBreak())
        
        return story

    def add_header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        canvas.saveState()
        
        # Header
        header_text = "UPS System Analysis - Appendix"
        canvas.setFont('Times-Roman', 10)
        canvas.drawString(self.margin, self.page_height - 0.5*inch, header_text)
        
        # Footer with page number
        page_num = canvas.getPageNumber()
        footer_text = f"Page {page_num}"
        canvas.drawRightString(self.page_width - self.margin, 0.5*inch, footer_text)
        
        # Draw header line
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(0.5)
        canvas.line(self.margin, self.page_height - 0.6*inch, 
                   self.page_width - self.margin, self.page_height - 0.6*inch)
        
        canvas.restoreState()

    def generate_pdf(self):
        """Generate the complete PDF appendix."""
        print("Starting PDF generation...")
        print(f"Output file: {self.output_filename}")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            self.output_filename,
            pagesize=A4,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=self.margin + 0.5*inch,  # Extra space for header
            bottomMargin=self.margin + 0.5*inch  # Extra space for footer
        )
        
        # Generate content
        story = self.create_pdf_content()
        
        # Build PDF with header/footer
        doc.build(story, onFirstPage=self.add_header_footer, onLaterPages=self.add_header_footer)
        
        print(f"PDF appendix generated successfully: {self.output_filename}")
        print(f"Total events processed: {len(self.events)}")
        
        return self.output_filename

def main():
    """Main function to generate the PDF appendix."""
    print("UPS Analysis PDF Appendix Generator")
    print("=" * 50)
    
    # Create generator instance
    generator = AppendixGenerator()
    
    # Generate PDF
    output_file = generator.generate_pdf()
    
    print(f"\nPDF appendix created: {output_file}")
    print("The appendix includes:")
    print("- Voltage waveform plots")
    print("- Current waveform plots") 
    print("- THD analysis plots")
    print("- Detailed THD data tables")
    print("\nFormatted for professional thesis appendix use.")

if __name__ == "__main__":
    main()
