Class Demo.Util
{

ClassMethod loadsql(file As %String) As %Status
{
    //ログは入力ファイルと同じディレクトリに置く
    set logfile=##class(%File).GetDirectory(file)_"sqlload-log.txt"
    set sql="LOAD SQL FROM FILE '"_file_"' DIALECT 'IRIS'"
    set sql=sql_" LOG TO FILE '"_logfile_"' IGNORE ERRORS"
    set stmt=##class(%SQL.Statement).%New()
    set tResult = ##class(%SQL.Statement).%ExecDirect(.tStatement,sql)
    if tResult.%SQLCODE<0 {
        write "Error::",tResult.%Message,!
    }
}
}