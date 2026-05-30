$word = New-Object -ComObject Word.Application
$word.Visible = $false

$doc1Path = "c:\Users\juanm\Documents\Proyecto Automatización\Documentos\Informe avances auto.docx"
$out1Path = "c:\Users\juanm\Documents\Proyecto Automatización\Documentos\informe1.txt"
$doc1 = $word.Documents.Open($doc1Path)
$doc1.Content.Text | Set-Content -Path $out1Path -Encoding UTF8
$doc1.Close($false)

$doc2Path = "c:\Users\juanm\Documents\Proyecto Automatización\Documentos\avances con el paso a seguir.docx"
$out2Path = "c:\Users\juanm\Documents\Proyecto Automatización\Documentos\informe2.txt"
$doc2 = $word.Documents.Open($doc2Path)
$doc2.Content.Text | Set-Content -Path $out2Path -Encoding UTF8
$doc2.Close($false)

$word.Quit()
Write-Host "Done extracting documents"
