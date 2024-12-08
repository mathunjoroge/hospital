<?php
session_start();
include('../connect.php');
$doctor =$_SESSION['SESS_FIRST_NAME'];
$a = $_POST['name'];
$b= date("Y-m-d", strtotime($_POST['age']));
$e= $_POST['sex']; 
$h = $_POST['number'];
$dept = $_POST['dept'];
$sql = "INSERT INTO patients (name,age,sex,opno) VALUES (:a,:b,:e,:h)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$a,':b'=>$b,':e'=>$e,':h'=>$h)); 
//record to visits table
$sql = "INSERT INTO visits (patient) VALUES (:h)";
$q = $db->prepare($sql);
$q->execute(array(':h'=>$h)); 
?>
<?php

$j = $_POST['number'];
$date=date('Y-m-d');
if (isset($_POST['fees'])) {
$fees = $_POST['fees'];
$List = implode(', ',$fees);
$result = $db->prepare("SELECT GROUP_CONCAT(amount) AS amount FROM  fees   WHERE fees_id IN($List)");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
 $feeamount=$row['amount'];
$amounts = explode (",", $feeamount);
foreach (array_combine($fees, $amounts) as $fee => $amount){
$sql = "INSERT INTO collection (fees_id,date,paid_by,amount) VALUES (:a,:b,:c,:d)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$fee,':b'=>$date,':c'=>$h,':d'=>$amount));

}
}
}
if (isset($_POST['lab'])) {
$labs = $_POST['lab'];
foreach ($labs as $lab) {
$sql = "INSERT INTO lab (test,opn,reqby) VALUES (:a,:b,:c)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $lab,
':b' => $h,
':c' => $doctor
));
}
}
// set the patient patients records to not paid

$served=1;
$sql ="UPDATE patients
      SET   has_bill=?,
      served=?
      WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$has_bill,$h));
// end setting
if ($dept==2) {
header("location: admit.php?search==$h&response=0");
}

else{
    $receipt=rand();
header("location: receipt.php?name=$a&number=$h&receipt=$receipt");
}
?>
