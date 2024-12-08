<?php
session_start();
include('../connect.php');
$a = date('Y-m-d');
if (isset($_POST['sys'])) {
	// code...
	$b = $_POST['sys'];
}
else{
	$b =='';
}
//check height
if (isset($_POST['height'])) {
	// code...
	$c = $_POST['height'];
}
else{
	$c='';
}
$d = $_POST['opno'];
//check dystolic
if (isset($_POST['dys'])) {
	// code...
	$e = $_POST['dys'];
}
else{
	$e='';
}
$f = $_POST['rate'];
//check weight
if (isset($_POST['weight'])) {
	// code...
	$g = $_POST['weight'];
}
else{
	$g='';
}
//check temp
if (isset($_POST['temp'])) {
	// code...
	$h = $_POST['temp'];
}
else{
	$h='';
}
//check breath rate
if (isset($_POST['br'])) {
	// code...
	$j = $_POST['br'];
}
else{
	$j='';
}
//check breath rbs
if (isset($_POST['rbs'])) {
	// code...
	$k = $_POST['rbs'];
}
else{
	$k='';
}

//check breath muac
if (isset($_POST['muac'])) {
	// code...
	$l = $_POST['muac'];
}
else{
	$l='';
}
$l = $_POST['muac'];
$spo = $_POST['spo']; 
$sql = "INSERT INTO vitals (date,systolic,height,pno,diastolic,rate,weight,temperature,breat_rate,rbs,muac,spo) VALUES (:a,:b,:c,:d,:e,:f,:g,:h,:j,:k,:l,:spo)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$a,':b'=>$b,':c'=>$c,':d'=>$d,':e'=>$e,':f'=>$f,':g'=>$g,':h'=>$h,':j'=>$j,':k'=>$k,':l'=>$l,':spo'=>$spo));
//post data to gyn table if is posted
if (isset($_POST['lmp'])) {
$lmp = date("Y-m-d", strtotime($_POST['lmp']));
$edd = date("Y-m-d", strtotime($_POST['edd']));
$para = $_POST['para'];
$gravid = $_POST['gravid'];
$live_births = $_POST['live_births'];
$births_alive = $_POST['births_alive'];
$sql = "INSERT INTO gyn (patient,lmp,edd,para,gravid,live_births,births_alive) VALUES (:a,:b,:c,:d,:e,:f,:g)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$d,':b'=>$lmp,':c'=>$edd,':d'=>$para,':e'=>$gravid,':f'=>$live_births,':g'=>$births_alive));
	
}
//remove patient from waiting list
$reset=2;
$sql = "UPDATE patients
        SET  served=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($reset,$d));

 $has_bill =1;
$sql = "UPDATE patients
        SET  has_bill=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$d)); 
header("location: index.php?search= &response=1");	
?>