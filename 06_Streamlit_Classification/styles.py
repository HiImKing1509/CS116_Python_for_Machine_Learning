class styles: 
	# Custom separate lines
	lines_separate_style = """ 
		<hr 
			style="height: 3px;
			border: one;
			color: #333;
			background-color: #333;" 
		/> 
	"""
 
	lines_section_separate_style = """ 
		<hr 
			style="height: 6px;
			border: none;
			color: #FFD700;
			background-color: #FFD700;" 
		/> 
	"""
 
def text_success(str):
    return f""" <style>p.sc{{color: Green; font-size: 16px; font-weight: 900;}}</style><p class="sc">{str} is uploaded</p> """