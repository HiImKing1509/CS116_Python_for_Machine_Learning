class styles:
    
    # Custom font page
	streamlit_style = """
		<style>
			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

			html, body, [class*="css"]  {
				font-family: 'Roboto', sans-serif;
			}
		</style>
	"""
 
	# Custom separate lines
	lines_separate_style = """ 
		<hr 
			style="height:3px;
			border:none;
			color:#333;
			background-color: #333;" 
		/> 
	"""
    
	def selected_style_checked(str):
		feature_selected = f"""
			<style>
				div > p.a {{
						color: White;
						font-size: 16px;
						font-weight: 900;
						margin: 20px;
						font-size: 16px;
						font-weight: bold;
					}}
				#containera {{
					border-radius: 5px;
					background-color: Green;
					margin: 4px;
					padding: 4px;
				}}
			</style> 
			<div id="containera">
				<p class="a">{str}</p>
			</div>
		"""
		return feature_selected
    
	def selected_style_unchecked(str):
		feature_unselected = f"""
			<style>
				div > p.b {{
						color: White;
						font-size: 16px;
						font-weight: 900;
						margin: 20px;
						font-size: 16px;
						font-weight: bold;
					}}
				#containerb {{
					border-radius: 5px;
					background-color: Red;
					margin: 4px;
					padding: 4px;
				}}
			</style> 
			<div id="containerb">
				<p class="b">{str}</p>
			</div>
		"""
		return feature_unselected