<?php
include('../connect.php');
session_start(); 
$a=strtok($_POST['drug'], '-');
$b = $_POST['freq'];
$c = $_POST['duration'];
$d = $_POST['pn'];
$e = $_POST['code'];
$f = $_POST['roa'];
$g = $_POST['strength']." ".$_POST['units'];
$doctor =$_SESSION['SESS_FIRST_NAME'];
$sql = "INSERT INTO prescribed_meds (drug,frequency,duration,patient,code,pres_by,roa,strength) VALUES (:a,:b,:c,:d,:e,:h,:f,:g)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$a,':b'=>$b,':c'=>$c,':d'=>$d,':e'=>$e,':h'=>$doctor,':f'=>$f,':g'=>$g));
 $has_bill =1;
$sql = "UPDATE patients
        SET  has_bill=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$d)); 
header("location:prescribe_inp.php?search=$d&response=0&code=$e");
?>