<?php
session_start();
$charge_ids= ($_POST['charge']);
$b = date('Y-m-d');
$c = $_POST['patient'];
$d="";
$e=$_SESSION['SESS_FIRST_NAME'];
$amounts= ($_POST['amount']);
include('../connect.php');
foreach (array_combine($charge_ids, $amounts) as $charge_id => $amount){
$sql = "INSERT INTO collection (fees_id, date, paid_by,paid,cashed_by,amount) VALUES(:a,:b,:c,:d,:e,:f)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$charge_id,':b'=>$b,':c'=>$c,':d'=>$d,':e'=>$e,':f'=>$amount)); 
}
 $has_bill =1;
$sql = "UPDATE patients
        SET  has_bill=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$c)); 
header("location:nursing.php?response=1");
?>
